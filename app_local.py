"""
AI Email Assistant with Local Authentication
Alternative main application file that uses local auth instead of Replit Auth
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from datetime import datetime
import json

# Import our modules
from models import db, User, Team, Email, EmailTemplate, AIModel, EmailTone, EmailStatus
from ai_service import AIService
from email_service import EmailService
from local_auth import LocalAuthService, setup_local_auth_routes
import routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local_email_assistant.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize services
ai_service = AIService()
email_service = EmailService()
local_auth = LocalAuthService(app)

# Setup authentication routes
setup_local_auth_routes(app)

# Create tables
with app.app_context():
    db.create_all()
    logger.info("Database tables created")

# Import and register routes (modify to use local auth)
@app.context_processor
def inject_current_user():
    """Inject current user into all templates"""
    return dict(current_user=local_auth.get_current_user())

@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, otherwise show login"""
    user = local_auth.get_current_user()
    if user:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@local_auth.require_login
def dashboard():
    """Main dashboard"""
    try:
        user = local_auth.get_current_user()
        
        # Get user's recent emails
        recent_emails = Email.query.filter_by(user_id=user.id).order_by(Email.created_at.desc()).limit(10).all()
        
        # Get user's templates
        templates = EmailTemplate.query.filter_by(user_id=user.id).limit(5).all()
        
        return render_template('dashboard.html',
                             user=user,
                             recent_emails=recent_emails,
                             templates=templates,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone))
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        flash('Error loading dashboard', 'error')
        return redirect(url_for('login'))

@app.route('/compose')
@local_auth.require_login
def compose():
    """Compose email page"""
    try:
        user = local_auth.get_current_user()
        templates = EmailTemplate.query.filter_by(user_id=user.id).all()
        
        return render_template('compose.html',
                             user=user,
                             templates=templates,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone))
    except Exception as e:
        logger.error(f"Error loading compose page: {str(e)}")
        flash('Error loading compose page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/settings')
@local_auth.require_login
def settings():
    """User settings page"""
    try:
        user = local_auth.get_current_user()
        
        # Get common SMTP providers for dropdown
        smtp_providers = {
            'gmail': email_service.get_common_smtp_settings('gmail'),
            'outlook': email_service.get_common_smtp_settings('outlook'),
            'yahoo': email_service.get_common_smtp_settings('yahoo'),
            'custom': email_service.get_common_smtp_settings('custom')
        }
        
        return render_template('settings.html',
                             user=user,
                             smtp_providers=smtp_providers,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone))
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        flash('Error loading settings', 'error')
        return redirect(url_for('dashboard'))

# API Routes with local auth
@app.route('/api/generate-reply', methods=['POST'])
@local_auth.require_login
def generate_reply():
    """Generate AI email reply"""
    try:
        data = request.get_json()
        
        original_email = data.get('original_email', '')
        context = data.get('context', '')
        tone = data.get('tone', 'professional')
        instructions = data.get('instructions', '')
        
        if not original_email:
            return jsonify({'success': False, 'error': 'Original email is required'}), 400
        
        result = ai_service.generate_email_reply_with_langchain(
            original_email=original_email,
            context=context,
            tone=tone,
            instructions=instructions
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating reply: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-sentiment', methods=['POST'])
@local_auth.require_login
def analyze_sentiment():
    """Analyze email sentiment using AI"""
    try:
        data = request.get_json()
        email_content = data.get('email_content', '')
        
        if not email_content:
            return jsonify({'success': False, 'error': 'Email content is required'}), 400
        
        result = ai_service.analyze_email_with_langchain(email_content)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/suggest-improvements', methods=['POST'])
@local_auth.require_login
def suggest_improvements():
    """Get AI suggestions for email improvements"""
    try:
        data = request.get_json()
        email_content = data.get('email_content', '')
        
        if not email_content:
            return jsonify({'success': False, 'error': 'Email content is required'}), 400
        
        result = ai_service.suggest_email_improvements(email_content)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-smtp-connection', methods=['POST'])
@local_auth.require_login
def test_smtp_connection():
    """Test SMTP connection with provided settings"""
    try:
        data = request.get_json()
        
        smtp_server = data.get('smtp_server', '').strip()
        smtp_port = data.get('smtp_port')
        smtp_username = data.get('smtp_username', '').strip()
        smtp_password = data.get('smtp_password', '').strip()
        smtp_use_tls = data.get('smtp_use_tls', True)
        
        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            return jsonify({'success': False, 'error': 'All SMTP fields are required'}), 400
        
        # Test connection
        result = email_service.test_smtp_connection(
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
            smtp_username=smtp_username,
            smtp_password=smtp_password,
            use_tls=smtp_use_tls
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error testing SMTP connection: {str(e)}")
        return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'}), 500

@app.route('/api/update-smtp-settings', methods=['POST'])
@local_auth.require_login
def update_smtp_settings():
    """Update user's SMTP settings"""
    try:
        data = request.get_json()
        user = local_auth.get_current_user()
        
        smtp_server = data.get('smtp_server', '').strip()
        smtp_port = data.get('smtp_port')
        smtp_username = data.get('smtp_username', '').strip()
        smtp_password = data.get('smtp_password', '').strip()
        smtp_use_tls = data.get('smtp_use_tls', True)
        
        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            return jsonify({'success': False, 'error': 'All SMTP fields are required'}), 400
        
        # Test connection first
        test_result = email_service.test_smtp_connection(
            smtp_server=smtp_server,
            smtp_port=int(smtp_port),
            smtp_username=smtp_username,
            smtp_password=smtp_password,
            use_tls=smtp_use_tls
        )
        
        if not test_result['success']:
            return jsonify(test_result), 400
        
        # Update user settings
        user.smtp_server = smtp_server
        user.smtp_port = int(smtp_port)
        user.smtp_username = smtp_username
        user.smtp_password = smtp_password  # Note: Should encrypt this in production
        user.smtp_use_tls = smtp_use_tls
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'SMTP settings updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating SMTP settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/templates')
@local_auth.require_login
def templates_page():
    """Email templates page"""
    try:
        user = local_auth.get_current_user()
        templates = EmailTemplate.query.filter_by(user_id=user.id).all()
        
        return render_template('templates.html',
                             user=user,
                             templates=templates,
                             ai_models=list(AIModel),
                             email_tones=list(EmailTone))
    except Exception as e:
        logger.error(f"Error loading templates page: {str(e)}")
        flash('Error loading templates page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/team')
@local_auth.require_login
def team():
    """Team page"""
    try:
        user = local_auth.get_current_user()
        return render_template('team.html', user=user)
    except Exception as e:
        logger.error(f"Error loading team page: {str(e)}")
        flash('Error loading team page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/analytics')
@local_auth.require_login
def analytics():
    """Analytics page"""
    try:
        user = local_auth.get_current_user()
        return render_template('analytics.html', user=user)
    except Exception as e:
        logger.error(f"Error loading analytics page: {str(e)}")
        flash('Error loading analytics page', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api_documentation')
@local_auth.require_login
def api_documentation():
    """API documentation page"""
    try:
        user = local_auth.get_current_user()
        return render_template('api_documentation.html', user=user)
    except Exception as e:
        logger.error(f"Error loading API documentation: {str(e)}")
        flash('Error loading API documentation', 'error')
        return redirect(url_for('dashboard'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    user = local_auth.get_current_user()
    if user:
        logger.info(f"User {user.email} connected via Socket.IO")
        emit('status', {'message': 'Connected to AI Email Assistant'})
    else:
        emit('error', {'message': 'Authentication required'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    user = local_auth.get_current_user()
    if user:
        logger.info(f"User {user.email} disconnected")

if __name__ == '__main__':
    logger.info("🚀 Starting AI Email Assistant with Local Authentication...")
    logger.info("📧 Application URL: http://localhost:5000")
    logger.info("🔐 Local Authentication: Enabled")
    
    # Run the application
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)