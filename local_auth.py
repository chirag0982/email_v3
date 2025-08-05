"""
Local Authentication System for AI Email Assistant
Alternative to Replit Auth for local development and self-hosted deployments
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, request, jsonify, redirect, url_for, render_template, flash
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User
import logging

class LocalAuthService:
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the local auth service with Flask app"""
        app.config.setdefault('SECRET_KEY', secrets.token_hex(32))
        app.config.setdefault('SESSION_PERMANENT', False)
        app.config.setdefault('PERMANENT_SESSION_LIFETIME', timedelta(hours=24))
    
    def hash_password(self, password: str) -> str:
        """Hash a password for storage"""
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password_hash: str, password: str) -> bool:
        """Check if password matches hash"""
        return check_password_hash(password_hash, password)
    
    def generate_user_id(self, email: str) -> str:
        """Generate a unique user ID from email"""
        return hashlib.sha256(email.encode()).hexdigest()[:16]
    
    def create_user(self, email: str, password: str, first_name: str = '', last_name: str = '') -> dict:
        """Create a new user account"""
        try:
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                return {'success': False, 'error': 'User already exists with this email'}
            
            # Create new user
            user_id = self.generate_user_id(email)
            password_hash = self.hash_password(password)
            
            user = User(
                id=user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                password_hash=password_hash,  # Add this field to User model
                created_at=datetime.now()
            )
            
            db.session.add(user)
            db.session.commit()
            
            logging.info(f"New user created: {email}")
            return {'success': True, 'user_id': user_id}
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error creating user: {e}")
            return {'success': False, 'error': str(e)}
    
    def authenticate_user(self, email: str, password: str) -> dict:
        """Authenticate user with email and password"""
        try:
            user = User.query.filter_by(email=email).first()
            if not user:
                return {'success': False, 'error': 'Invalid email or password'}
            
            if not hasattr(user, 'password_hash') or not user.password_hash:
                return {'success': False, 'error': 'Account not set up for local login'}
            
            if not self.check_password(user.password_hash, password):
                return {'success': False, 'error': 'Invalid email or password'}
            
            # Create session
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['logged_in'] = True
            session.permanent = True
            
            logging.info(f"User logged in: {email}")
            return {'success': True, 'user': user}
            
        except Exception as e:
            logging.error(f"Error authenticating user: {e}")
            return {'success': False, 'error': 'Authentication failed'}
    
    def logout_user(self):
        """Log out current user"""
        session.clear()
        return {'success': True}
    
    def get_current_user(self):
        """Get current logged-in user"""
        if not session.get('logged_in') or not session.get('user_id'):
            return None
        
        try:
            user = User.query.get(session['user_id'])
            return user
        except Exception as e:
            logging.error(f"Error getting current user: {e}")
            return None
    
    def require_login(self, f):
        """Decorator to require login for routes"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('logged_in') or not session.get('user_id'):
                if request.is_json:
                    return jsonify({'success': False, 'error': 'Login required'}), 401
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function

# Global instance
local_auth = LocalAuthService()

def setup_local_auth_routes(app):
    """Set up authentication routes"""
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'GET':
            return render_template('auth/login.html')
        
        try:
            data = request.get_json() if request.is_json else request.form
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            if not email or not password:
                error = 'Email and password are required'
                if request.is_json:
                    return jsonify({'success': False, 'error': error}), 400
                flash(error, 'error')
                return render_template('auth/login.html')
            
            result = local_auth.authenticate_user(email, password)
            
            if result['success']:
                if request.is_json:
                    return jsonify({'success': True, 'redirect': url_for('dashboard')})
                return redirect(url_for('dashboard'))
            else:
                if request.is_json:
                    return jsonify(result), 401
                flash(result['error'], 'error')
                return render_template('auth/login.html')
                
        except Exception as e:
            logging.error(f"Login error: {e}")
            error = 'Login failed'
            if request.is_json:
                return jsonify({'success': False, 'error': error}), 500
            flash(error, 'error')
            return render_template('auth/login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'GET':
            return render_template('auth/register.html')
        
        try:
            data = request.get_json() if request.is_json else request.form
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            confirm_password = data.get('confirm_password', '')
            first_name = data.get('first_name', '').strip()
            last_name = data.get('last_name', '').strip()
            
            # Validation
            if not email or not password:
                error = 'Email and password are required'
                if request.is_json:
                    return jsonify({'success': False, 'error': error}), 400
                flash(error, 'error')
                return render_template('auth/register.html')
            
            if len(password) < 6:
                error = 'Password must be at least 6 characters long'
                if request.is_json:
                    return jsonify({'success': False, 'error': error}), 400
                flash(error, 'error')
                return render_template('auth/register.html')
            
            if password != confirm_password:
                error = 'Passwords do not match'
                if request.is_json:
                    return jsonify({'success': False, 'error': error}), 400
                flash(error, 'error')
                return render_template('auth/register.html')
            
            result = local_auth.create_user(email, password, first_name, last_name)
            
            if result['success']:
                # Auto-login after registration
                auth_result = local_auth.authenticate_user(email, password)
                if auth_result['success']:
                    if request.is_json:
                        return jsonify({'success': True, 'redirect': url_for('dashboard')})
                    flash('Account created successfully!', 'success')
                    return redirect(url_for('dashboard'))
            
            if request.is_json:
                return jsonify(result), 400
            flash(result['error'], 'error')
            return render_template('auth/register.html')
            
        except Exception as e:
            logging.error(f"Registration error: {e}")
            error = 'Registration failed'
            if request.is_json:
                return jsonify({'success': False, 'error': error}), 500
            flash(error, 'error')
            return render_template('auth/register.html')
    
    @app.route('/logout')
    def logout():
        local_auth.logout_user()
        flash('You have been logged out', 'info')
        return redirect(url_for('login'))
    
    @app.route('/profile')
    @local_auth.require_login
    def profile():
        user = local_auth.get_current_user()
        return render_template('auth/profile.html', user=user)