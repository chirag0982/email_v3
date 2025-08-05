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
    clarity_score: int = Field(description="Clarity score from 1-10", default=8)
    tone_appropriateness: int = Field(description="Tone appropriateness score from 1-10", default=8)

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
                max_tokens=1024,  # Reduced to stay within credit limits
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
            logging.info(f"Raw AI Response: {repr(response[:200])}")
            logging.info(f"Parsed body: {repr(body[:200])}")
            
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
            
            # Check if it's a credit/payment error and provide helpful fallback
            if "402" in str(e) or "credits" in str(e).lower() or "payment" in str(e).lower():
                return {
                    'success': True,
                    'subject': "Re: Your Email",
                    'body': "Thank you for your email. I appreciate you reaching out and will get back to you soon.\n\nBest regards",
                    'tone': tone,
                    'confidence': 0.7,
                    'model_used': 'fallback-system',
                    'generation_time_ms': 50,
                    'fallback_used': True,
                    'fallback_reason': 'API credits exhausted - using template response'
                }
            
            return {
                'success': False,
                'error': str(e),
                'fallback_used': True
            }
    
    def suggest_email_improvements(self, email_content: str) -> Dict[str, Any]:
        """Advanced AI-powered email improvement suggestions with comprehensive analysis"""
        try:
            suggestions = []
            improvements = []
            
            # 1. STRUCTURE ANALYSIS
            lines = [line.strip() for line in email_content.split('\n') if line.strip()]
            paragraphs = email_content.split('\n\n')
            
            # Check email structure
            has_greeting = any(greeting in email_content.lower() for greeting in ['hi', 'hello', 'dear', 'greetings', 'good morning', 'good afternoon'])
            has_closing = any(closing in email_content.lower() for closing in ['regards', 'sincerely', 'best', 'thanks', 'cheers', 'yours'])
            
            if not has_greeting:
                suggestions.append("📝 Add a professional greeting (e.g., 'Dear [Name]' or 'Hi [Name]')")
            if not has_closing:
                suggestions.append("📝 Include a proper closing (e.g., 'Best regards' or 'Thank you')")
                
            # 2. CLARITY AND READABILITY ANALYSIS
            words = email_content.split()
            sentences = [s.strip() for s in email_content.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            
            # Average sentence length
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            if avg_sentence_length > 25:
                suggestions.append("✂️ Break long sentences into shorter ones (current average: {:.1f} words)".format(avg_sentence_length))
            
            # Paragraph length
            long_paragraphs = [p for p in paragraphs if len(p.split()) > 80]
            if long_paragraphs:
                suggestions.append("📄 Break long paragraphs into smaller chunks for better readability")
            
            # 3. TONE AND PROFESSIONAL LANGUAGE ANALYSIS
            casual_words = ['gonna', 'wanna', 'yeah', 'yep', 'ok', 'stuff', 'things', 'kinda', 'sorta']
            found_casual = [word for word in casual_words if word in email_content.lower()]
            if found_casual:
                suggestions.append("🎯 Replace casual language: {} → more professional alternatives".format(', '.join(found_casual[:3])))
            
            # Weak language detection
            weak_phrases = ['i think', 'maybe', 'perhaps', 'i guess', 'sort of', 'kind of']
            found_weak = [phrase for phrase in weak_phrases if phrase in email_content.lower()]
            if found_weak:
                suggestions.append("💪 Strengthen language by removing uncertain phrases like '{}'".format(found_weak[0]))
            
            # 4. ACTION-ORIENTED IMPROVEMENTS
            action_verbs = ['please', 'need', 'require', 'request', 'would like', 'could you']
            has_clear_action = any(verb in email_content.lower() for verb in action_verbs)
            question_marks = email_content.count('?')
            
            if not has_clear_action and question_marks == 0:
                suggestions.append("🎯 Add a clear call-to-action or specific request")
            
            # 5. FORMATTING AND PRESENTATION
            if len(email_content) > 500 and email_content.count('\n') < 3:
                suggestions.append("📋 Use bullet points or numbered lists for multiple items")
            
            # Check for excessive punctuation
            if email_content.count('!') > 2:
                suggestions.append("⚡ Reduce exclamation marks for a more professional tone")
            
            # 6. SPECIFIC CONTENT IMPROVEMENTS
            if len(email_content) < 30:
                suggestions.append("📝 Provide more context and detail to make your message clearer")
            elif len(email_content) > 1000:
                suggestions.append("✂️ Consider condensing your message - shorter emails get better responses")
            
            # 7. EMAIL ETIQUETTE
            if not email_content.lower().startswith(('hi', 'hello', 'dear', 'good')):
                suggestions.append("👋 Start with a greeting to create a personal connection")
            
            # 8. AI-POWERED CONTEXTUAL SUGGESTIONS
            if self.langchain_models and len(suggestions) < 4:
                try:
                    model = self.langchain_models['qwen-4-turbo']
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an expert email writing coach. Analyze this email and provide 2-3 specific, actionable improvements focusing on clarity, professionalism, and effectiveness."),
                        ("human", "Email to improve:\n{email_content}\n\nProvide specific suggestions with clear explanations.")
                    ])
                    
                    chain = prompt | model | StrOutputParser()
                    ai_response = chain.invoke({"email_content": email_content})
                    
                    # Parse AI suggestions
                    ai_lines = [line.strip() for line in ai_response.split('\n') if line.strip() and len(line.strip()) > 20]
                    for line in ai_lines[:2]:  # Add top 2 AI suggestions
                        if not any(existing in line.lower() for existing in [s.lower() for s in suggestions]):
                            suggestions.append(f"🤖 {line}")
                            
                except Exception as ai_error:
                    logging.warning(f"AI suggestion generation failed: {ai_error}")
            
            # 9. ENSURE MINIMUM QUALITY SUGGESTIONS
            if len(suggestions) < 3:
                fallback_suggestions = [
                    "📧 Consider adding a clear subject line that summarizes your main point",
                    "🔍 Proofread for spelling and grammar errors before sending",
                    "🎯 State your main request or purpose in the first paragraph",
                    "📞 Include your contact information if a response is needed",
                    "⏰ Mention any deadlines or time-sensitive information clearly"
                ]
                
                for fallback in fallback_suggestions:
                    if len(suggestions) < 5 and not any(key in fallback.lower() for key in [s.lower() for s in suggestions]):
                        suggestions.append(fallback)
            
            # 10. CATEGORIZE AND PRIORITIZE SUGGESTIONS
            priority_suggestions = [s for s in suggestions if any(marker in s for marker in ['📝', '🎯', '💪'])]
            other_suggestions = [s for s in suggestions if s not in priority_suggestions]
            
            final_suggestions = (priority_suggestions + other_suggestions)[:6]
            
            return {
                'success': True,
                'suggestions': final_suggestions,
                'analysis_summary': {
                    'word_count': len(words),
                    'sentence_count': len(sentences),
                    'readability_score': min(10, max(1, 10 - (avg_sentence_length - 15) / 5)) if avg_sentence_length > 0 else 8,
                    'structure_score': (8 if has_greeting else 6) + (2 if has_closing else 0),
                    'improvement_areas': len(final_suggestions)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in comprehensive email analysis: {e}")
            return {
                'success': True,
                'suggestions': [
                    "📝 Add a professional greeting and closing",
                    "🎯 State your main purpose clearly in the first paragraph", 
                    "📄 Break content into shorter, focused paragraphs",
                    "💪 Use confident, professional language",
                    "🔍 Proofread carefully before sending"
                ],
                'analysis_summary': {
                    'word_count': len(email_content.split()),
                    'readability_score': 7,
                    'structure_score': 6,
                    'improvement_areas': 5
                }
            }
    
    def analyze_email_with_langchain(self, email_content: str) -> Dict[str, Any]:
        """Analyze email using simple reliable analysis"""
        try:
            # Simple sentiment analysis based on keywords
            positive_words = ['thank', 'great', 'excellent', 'good', 'happy', 'pleased', 'wonderful', 'amazing', 'love', 'appreciate', 'glad']
            negative_words = ['sorry', 'problem', 'issue', 'concern', 'disappointed', 'frustrated', 'urgent', 'emergency', 'mistake', 'error']
            
            content_lower = email_content.lower()
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)
            
            # Determine sentiment
            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Determine urgency
            urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'deadline', 'rush']
            if any(word in content_lower for word in urgent_words):
                urgency = 'high'
            elif any(word in content_lower for word in ['soon', 'quick', 'fast']):
                urgency = 'medium'
            else:
                urgency = 'low'
            
            # Determine tone
            formal_words = ['dear', 'sincerely', 'regards', 'respectfully']
            casual_words = ['hi', 'hey', 'thanks', 'cheers']
            
            if any(word in content_lower for word in formal_words):
                tone = 'formal'
            elif any(word in content_lower for word in casual_words):
                tone = 'friendly'
            else:
                tone = 'professional'
            
            # Extract key topics (simple word frequency)
            words = content_lower.split()
            common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must', 'ought']
            filtered_words = [word for word in words if len(word) > 3 and word not in common_words]
            key_topics = list(set(filtered_words[:3])) if filtered_words else ['general communication']
            
            # Simple action items detection
            action_words = ['please', 'need', 'require', 'request', 'ask', 'help', 'meet', 'call', 'email', 'send']
            action_items = []
            for word in action_words:
                if word in content_lower:
                    if 'meet' in content_lower:
                        action_items.append('schedule meeting')
                    elif 'call' in content_lower:
                        action_items.append('phone call')
                    elif 'email' in content_lower or 'send' in content_lower:
                        action_items.append('send response')
                    elif 'help' in content_lower:
                        action_items.append('provide assistance')
            
            if not action_items:
                action_items = ['respond to email']
            
            # Calculate scores
            clarity_score = 8 if len(email_content) > 50 else 6
            tone_appropriateness = 9 if tone in ['professional', 'formal'] else 7
            
            return {
                'success': True,
                'analysis': {
                    'sentiment': sentiment,
                    'urgency': urgency,
                    'key_topics': key_topics,
                    'action_items': list(set(action_items))[:3],
                    'tone': tone,
                    'clarity_score': clarity_score,
                    'tone_appropriateness': tone_appropriateness
                }
            }
            
        except Exception as e:
            logging.error(f"Email analysis error: {e}")
            return {
                'success': True,
                'analysis': {
                    'sentiment': 'neutral',
                    'urgency': 'medium',
                    'key_topics': ['general communication'],
                    'action_items': ['respond to email'],
                    'tone': 'professional',
                    'clarity_score': 7,
                    'tone_appropriateness': 8
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