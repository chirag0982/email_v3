import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from models import Email, EmailStatus, User, Team
from app import db

class EmailService:
    def __init__(self):
        self.smtp_connections = {}  # Connection pooling
    
    def send_email(self, 
                   user: User,
                   to_addresses: List[str],
                   subject: str,
                   body_html: str,
                   body_text: str = None,
                   cc_addresses: List[str] = None,
                   bcc_addresses: List[str] = None,
                   email_id: str = None) -> Dict[str, Any]:
        """
        Send an email using the user's SMTP configuration
        """
        try:
            # Validate user has SMTP configuration
            if not all([user.smtp_server, user.smtp_port, user.smtp_username, user.smtp_password]):
                return {
                    'success': False,
                    'error': 'SMTP configuration not complete. Please configure your email settings.'
                }
            
            # Create email message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = user.email or user.smtp_username
            message['To'] = ', '.join(to_addresses)
            
            if cc_addresses:
                message['Cc'] = ', '.join(cc_addresses)
            
            # Add body content
            if body_text:
                text_part = MIMEText(body_text, 'plain')
                message.attach(text_part)
            
            if body_html:
                html_part = MIMEText(body_html, 'html')
                message.attach(html_part)
            
            # Prepare recipient list
            recipients = to_addresses[:]
            if cc_addresses:
                recipients.extend(cc_addresses)
            if bcc_addresses:
                recipients.extend(bcc_addresses)
            
            # Send email
            result = self._send_smtp_email(user, message, recipients)
            
            # Update email status in database if email_id provided
            if email_id and result['success']:
                email = Email.query.get(email_id)
                if email:
                    email.status = EmailStatus.SENT
                    email.sent_at = datetime.now()
                    db.session.commit()
            
            return result
            
        except Exception as e:
            logging.error(f"Error sending email: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to send email: {str(e)}'
            }
    
    def _send_smtp_email(self, user: User, message: MIMEMultipart, recipients: List[str]) -> Dict[str, Any]:
        """
        Send email via SMTP
        """
        try:
            # Create SMTP connection
            if user.smtp_use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(user.smtp_server, user.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP_SSL(user.smtp_server, user.smtp_port)
            
            # Login to SMTP server
            server.login(user.smtp_username, user.smtp_password)
            
            # Send email
            text = message.as_string()
            server.sendmail(user.smtp_username, recipients, text)
            server.quit()
            
            return {
                'success': True,
                'message': f'Email sent successfully to {len(recipients)} recipients'
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                'success': False,
                'error': 'SMTP authentication failed. Please check your email credentials.'
            }
        except smtplib.SMTPRecipientsRefused as e:
            return {
                'success': False,
                'error': f'Recipients refused: {str(e)}'
            }
        except smtplib.SMTPServerDisconnected:
            return {
                'success': False,
                'error': 'SMTP server disconnected unexpectedly'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'SMTP error: {str(e)}'
            }
    
    def test_smtp_connection(self, smtp_server: str, smtp_port: int, 
                           smtp_username: str, smtp_password: str, 
                           use_tls: bool = True) -> Dict[str, Any]:
        """
        Test SMTP connection with provided credentials
        """
        try:
            if use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            
            server.login(smtp_username, smtp_password)
            server.quit()
            
            return {
                'success': True,
                'message': 'SMTP connection successful'
            }
            
        except smtplib.SMTPAuthenticationError:
            return {
                'success': False,
                'error': 'Authentication failed. Please check your credentials.'
            }
        except smtplib.SMTPConnectError:
            return {
                'success': False,
                'error': 'Could not connect to SMTP server. Please check server and port.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Connection test failed: {str(e)}'
            }
    
    def get_common_smtp_settings(self, provider: str) -> Dict[str, Any]:
        """
        Get common SMTP settings for popular email providers
        """
        settings = {
            'gmail': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True,
                'instructions': 'Use your Gmail address and an App Password (not your regular password)'
            },
            'outlook': {
                'smtp_server': 'smtp-mail.outlook.com',
                'smtp_port': 587,
                'use_tls': True,
                'instructions': 'Use your Outlook/Hotmail address and password'
            },
            'yahoo': {
                'smtp_server': 'smtp.mail.yahoo.com',
                'smtp_port': 587,
                'use_tls': True,
                'instructions': 'Use your Yahoo address and an App Password'
            },
            'custom': {
                'smtp_server': '',
                'smtp_port': 587,
                'use_tls': True,
                'instructions': 'Enter your custom SMTP server details'
            }
        }
        
        return settings.get(provider, settings['custom'])
    
    def schedule_email(self, email_id: str, send_time: datetime) -> Dict[str, Any]:
        """
        Schedule an email to be sent at a specific time
        """
        try:
            email = Email.query.get(email_id)
            if not email:
                return {
                    'success': False,
                    'error': 'Email not found'
                }
            
            email.scheduled_send_time = send_time
            email.status = EmailStatus.DRAFT  # Keep as draft until sent
            db.session.commit()
            
            return {
                'success': True,
                'message': f'Email scheduled for {send_time.strftime("%Y-%m-%d %H:%M")}'
            }
            
        except Exception as e:
            logging.error(f"Error scheduling email: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to schedule email: {str(e)}'
            }
    
    def get_email_analytics(self, team_id: str = None, user_id: str = None, 
                           start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        Get email analytics for a team or user
        """
        try:
            query = Email.query
            
            if team_id:
                query = query.filter_by(team_id=team_id)
            
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            if start_date:
                query = query.filter(Email.created_at >= start_date)
            
            if end_date:
                query = query.filter(Email.created_at <= end_date)
            
            emails = query.all()
            
            # Calculate analytics
            total_emails = len(emails)
            sent_emails = len([e for e in emails if e.status in [EmailStatus.SENT, EmailStatus.DELIVERED, EmailStatus.OPENED, EmailStatus.REPLIED]])
            draft_emails = len([e for e in emails if e.status == EmailStatus.DRAFT])
            failed_emails = len([e for e in emails if e.status == EmailStatus.FAILED])
            
            # AI model usage
            model_usage = {}
            for email in emails:
                if email.ai_model_used:
                    model = email.ai_model_used.value
                    model_usage[model] = model_usage.get(model, 0) + 1
            
            # Average generation time
            generation_times = [e.generation_time_ms for e in emails if e.generation_time_ms]
            avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
            
            # User ratings
            ratings = [e.user_rating for e in emails if e.user_rating]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            return {
                'success': True,
                'analytics': {
                    'total_emails': total_emails,
                    'sent_emails': sent_emails,
                    'draft_emails': draft_emails,
                    'failed_emails': failed_emails,
                    'success_rate': (sent_emails / total_emails * 100) if total_emails > 0 else 0,
                    'model_usage': model_usage,
                    'avg_generation_time_ms': round(avg_generation_time),
                    'avg_user_rating': round(avg_rating, 2) if avg_rating > 0 else None
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting email analytics: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to get analytics: {str(e)}'
            }

# Global email service instance
email_service = EmailService()
