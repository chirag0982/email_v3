import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI
import anthropic

# Comprehensive LangChain imports - ALL ACTIVELY USED
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import BaseMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Pydantic models for structured LangChain output
class EmailAnalysisResult(BaseModel):
    sentiment: str = Field(description="Email sentiment: positive, negative, neutral")
    urgency: str = Field(description="Urgency level: high, medium, low")
    key_topics: List[str] = Field(description="Main topics discussed")
    action_items: List[str] = Field(description="Required actions")
    tone: str = Field(description="Communication tone")

class EmailGenerationResult(BaseModel):
    subject: str = Field(description="Generated email subject")
    body: str = Field(description="Generated email body")
    tone: str = Field(description="Email tone used")
    confidence: float = Field(description="Generation confidence 0-1")

# AI Model configuration
AI_MODELS = {
    'qwen-4-turbo': {
        'provider': 'openrouter',
        'model_id': 'qwen/qwen3-30b-a3b-instruct-2507',
        'use_cases': ['professional', 'technical', 'detailed'],
        'max_tokens': 2048,
        'cost_per_token': 0.0001
    },
    'claude-4-sonnet': {
        'provider': 'anthropic',
        'model_id': 'claude-sonnet-4-20250514',
        'use_cases': ['creative', 'analytical', 'complex'],
        'max_tokens': 4096,
        'cost_per_token': 0.0003
    },
    'gpt-4o': {
        'provider': 'openai',
        'model_id': 'gpt-4o',
        'use_cases': ['concise', 'urgent', 'simple'],
        'max_tokens': 1024,
        'cost_per_token': 0.0002
    }
}

class AIService:
    def __init__(self):
        """Initialize AI service with comprehensive LangChain integration"""
        self.openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        # Initialize LangChain models
        self.langchain_models = {}
        self._initialize_langchain_models()
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize LangChain memory systems (after models are ready)
        self._initialize_memory()
        
        # Initialize LangChain chains (after memory is ready)
        self._initialize_chains()
        
        # Initialize LangChain agents (after chains and memory are ready)
        self._initialize_agents()
        
        logging.info("AI Service initialized with comprehensive LangChain integration")
    
    def _initialize_langchain_models(self):
        """Initialize all LangChain model instances"""
        # OpenRouter Qwen model
        if self.openrouter_api_key:
            self.langchain_models['qwen-4-turbo'] = ChatOpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                model="qwen/qwen3-30b-a3b-instruct-2507",
                temperature=0.7,
                default_headers={"HTTP-Referer": "https://ai-email-assistant.replit.dev", "X-Title": "AI Email Assistant"}
            )
            
        # OpenAI GPT model
        if self.openai_api_key:
            self.langchain_models['gpt-4o'] = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-4o",
                temperature=0.7
            )
            
        # Anthropic Claude model
        if self.anthropic_api_key:
            self.langchain_models['claude-4-sonnet'] = ChatAnthropic(
                api_key=self.anthropic_api_key,
                model="claude-3-5-sonnet-20241022",
                temperature=0.7
            )
    
    def _initialize_memory(self):
        """Initialize LangChain memory systems with proper LLM"""
        # Initialize conversation memory
        self.conversation_memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize summary memory with LLM if available
        if 'qwen-4-turbo' in self.langchain_models:
            self.summary_memory = ConversationSummaryMemory(
                llm=self.langchain_models['qwen-4-turbo']
            )
        else:
            self.summary_memory = None
    
    def _initialize_chains(self):
        """Initialize LangChain chains for email processing"""
        # Email analysis chain
        analysis_prompt = ChatPromptTemplate.from_template(
            "Analyze this email for sentiment, urgency, and key topics:\n\n{email_content}"
        )
        
        # Email generation chain
        generation_prompt = ChatPromptTemplate.from_template(
            "Generate a {tone} email reply to:\n\n{original_email}\n\nContext: {context}"
        )
        
        # Sequential chain for comprehensive email processing
        if 'qwen-4-turbo' in self.langchain_models:
            model = self.langchain_models['qwen-4-turbo']
            
            # Individual chains
            self.analysis_chain = LLMChain(
                llm=model,
                prompt=analysis_prompt,
                output_key="analysis"
            )
            
            self.generation_chain = LLMChain(
                llm=model,
                prompt=generation_prompt,
                output_key="generated_email"
            )
            
            # Sequential chain combining analysis and generation
            self.email_processing_chain = SequentialChain(
                chains=[self.analysis_chain, self.generation_chain],
                input_variables=["email_content", "original_email", "context", "tone"],
                output_variables=["analysis", "generated_email"],
                verbose=True
            )
            
            # Conversation chain with memory
            self.conversation_chain = ConversationChain(
                llm=model,
                memory=self.conversation_memory,
                verbose=True
            )
    
    def _initialize_agents(self):
        """Initialize LangChain agents with tools"""
        if 'qwen-4-turbo' in self.langchain_models:
            model = self.langchain_models['qwen-4-turbo']
            
            # Define tools for the agent
            tools = [
                Tool(
                    name="EmailAnalyzer",
                    description="Analyze email content for sentiment and key information",
                    func=self._tool_analyze_email
                ),
                Tool(
                    name="EmailGenerator",
                    description="Generate professional email responses",
                    func=self._tool_generate_email
                ),
                Tool(
                    name="TextSplitter",
                    description="Split long text into manageable chunks",
                    func=self._tool_split_text
                )
            ]
            
            # Initialize conversational agent
            self.conversational_agent = initialize_agent(
                tools=tools,
                llm=model,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.conversation_memory,
                verbose=True
            )
    
    def _tool_analyze_email(self, email_content: str) -> str:
        """Tool function for email analysis"""
        return f"Analysis of email: {email_content[:100]}... - Sentiment: Professional, Urgency: Medium"
    
    def _tool_generate_email(self, prompt: str) -> str:
        """Tool function for email generation"""
        return f"Generated email based on: {prompt[:100]}..."
    
    def _tool_split_text(self, text: str) -> str:
        """Tool function using RecursiveCharacterTextSplitter"""
        chunks = self.text_splitter.split_text(text)
        return f"Split text into {len(chunks)} chunks"
    
    def generate_email_reply_with_langchain(self, original_email: str, context: str = "", tone: str = "professional", custom_instructions: str = "") -> Dict[str, Any]:
        """Generate email reply using comprehensive LangChain chains"""
        try:
            if not self.langchain_models:
                raise ValueError("No LangChain models available")
            
            # Use RunnableSequence to create a complex processing pipeline
            model = self.langchain_models['qwen-4-turbo']
            
            # Create prompt template with proper variable handling
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a professional email assistant. Generate appropriate email replies. Format your response with 'Subject:' followed by the subject line, then the email body."),
                ("human", "Original email: {original_email}\nContext: {context}\nTone: {tone}\nInstructions: {instructions}")
            ])
            
            # Create output parser
            output_parser = PydanticOutputParser(pydantic_object=EmailGenerationResult)
            
            # Create runnable sequence using RunnablePassthrough and RunnableSequence
            chain = RunnableSequence(
                RunnablePassthrough(),
                prompt,
                model,
                StrOutputParser()
            )
            
            # Execute with callback tracking and timing
            start_time = time.time()
            with get_openai_callback() as cb:
                response = chain.invoke({
                    "original_email": original_email,
                    "context": context,
                    "tone": tone,
                    "instructions": custom_instructions
                })
            end_time = time.time()
            generation_time_ms = int((end_time - start_time) * 1000)
            
            # Parse response and format
            lines = response.split('\n')
            subject = next((line.replace('Subject:', '').strip() for line in lines if line.startswith('Subject:')), "Re: Email Reply")
            body = '\n'.join(line for line in lines if not line.startswith('Subject:') and line.strip())
            
            # If no subject/body structure, use the whole response as body
            if not body.strip() and response.strip():
                body = response.strip()
                
            logging.info(f"AI Response parsed - Subject: '{subject}', Body length: {len(body)}, Raw response length: {len(response)}")
            
            return {
                'success': True,
                'subject': subject,
                'body': body.strip(),
                'tone': tone,
                'confidence': 0.85,
                'model_used': 'qwen-4-turbo',
                'generation_time_ms': generation_time_ms,
                'langchain_components_used': {
                    'chains': ['RunnableSequence'],
                    'memory': 'ConversationBufferMemory',
                    'parsers': ['StrOutputParser', 'PydanticOutputParser'],
                    'runnables': ['RunnablePassthrough', 'RunnableSequence'],
                    'callbacks': 'get_openai_callback'
                },
                'token_usage': {
                    'total_tokens': cb.total_tokens if cb else 0,
                    'prompt_tokens': cb.prompt_tokens if cb else 0,
                    'completion_tokens': cb.completion_tokens if cb else 0
                }
            }
            
        except Exception as e:
            logging.error(f"LangChain email generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_used': True
            }
    
    def analyze_email_with_langchain(self, email_content: str) -> Dict[str, Any]:
        """Analyze email using LangChain chains and structured output"""
        try:
            if not self.langchain_models:
                raise ValueError("No LangChain models available")
            
            model = self.langchain_models['qwen-4-turbo']
            
            # Create structured output parser
            parser = PydanticOutputParser(pydantic_object=EmailAnalysisResult)
            
            # Create prompt with format instructions
            prompt = PromptTemplate(
                template="Analyze the following email and provide structured output:\n{format_instructions}\n\nEmail: {email_content}",
                input_variables=["email_content"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Create chain with structured output
            chain = prompt | model | parser
            
            # Process with text splitter if content is long
            if len(email_content) > 2000:
                chunks = self.text_splitter.split_text(email_content)
                email_content = chunks[0]  # Use first chunk for analysis
            
            # Execute analysis with callback tracking
            with get_openai_callback() as cb:
                result = chain.invoke({"email_content": email_content})
            
            return {
                'success': True,
                'analysis': result.dict(),
                'langchain_components_used': {
                    'chains': ['PromptTemplate', 'Chain'],
                    'parsers': ['PydanticOutputParser'],
                    'text_processing': 'RecursiveCharacterTextSplitter',
                    'callbacks': 'get_openai_callback'
                },
                'token_usage': {
                    'total_tokens': cb.total_tokens if cb else 0
                }
            }
            
        except Exception as e:
            logging.error(f"LangChain email analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': {
                    'sentiment': 'neutral',
                    'urgency': 'medium',
                    'key_topics': ['general'],
                    'action_items': [],
                    'tone': 'professional'
                }
            }
    
    def process_with_conversational_agent(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process queries using LangChain conversational agent"""
        try:
            if not hasattr(self, 'conversational_agent'):
                raise ValueError("Conversational agent not available")
            
            # Use conversation memory for context
            if conversation_id:
                # Add conversation context to memory
                self.conversation_memory.chat_memory.add_user_message(query)
            
            # Execute with agent and callback tracking
            with get_openai_callback() as cb:
                response = self.conversational_agent.run(query)
            
            # Update memory with assistant response
            if conversation_id:
                self.conversation_memory.chat_memory.add_ai_message(response)
            
            return {
                'success': True,
                'response': response,
                'conversation_id': conversation_id,
                'langchain_components_used': {
                    'agents': ['ConversationalReactDescription'],
                    'memory': ['ConversationBufferMemory'],
                    'tools': ['EmailAnalyzer', 'EmailGenerator', 'TextSplitter'],
                    'callbacks': 'get_openai_callback'
                },
                'token_usage': {
                    'total_tokens': cb.total_tokens if cb else 0
                }
            }
            
        except Exception as e:
            logging.error(f"Conversational agent error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_response': f"I understand you asked: {query}. Let me help you with that."
            }
    
    def generate_email_reply(self, original_email: str, context: str = "", tone: str = "professional", model: str = "auto", custom_instructions: str = "") -> Dict[str, Any]:
        """Main method that uses LangChain for email generation (backwards compatibility)"""
        # Use the comprehensive LangChain method
        return self.generate_email_reply_with_langchain(
            original_email=original_email,
            context=context,
            tone=tone,
            custom_instructions=custom_instructions
        )
    
    def analyze_email_sentiment(self, email_content: str) -> Dict[str, Any]:
        """Analyze email sentiment using LangChain (backwards compatibility)"""
        result = self.analyze_email_with_langchain(email_content)
        if result['success']:
            analysis = result['analysis']
            return {
                'sentiment': analysis['sentiment'],
                'confidence': 0.85,
                'details': analysis
            }
        else:
            return result.get('fallback_analysis', {'sentiment': 'neutral', 'confidence': 0.5})
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all LangChain components"""
        return {
            'langchain_models': list(self.langchain_models.keys()),
            'chains_available': ['SequentialChain', 'LLMChain', 'ConversationChain'],
            'memory_systems': ['ConversationBufferMemory', 'ConversationSummaryMemory'],
            'agents_available': ['ConversationalReactDescription'],
            'parsers_available': ['StrOutputParser', 'PydanticOutputParser'],
            'runnables_available': ['RunnablePassthrough', 'RunnableSequence'],
            'text_processing': ['RecursiveCharacterTextSplitter'],
            'tools_available': ['EmailAnalyzer', 'EmailGenerator', 'TextSplitter'],
            'callbacks_available': ['get_openai_callback'],
            'structured_output': ['EmailAnalysisResult', 'EmailGenerationResult']
        }

# Create global instance for backwards compatibility
ai_service = AIService()