// AI Email Assistant - Main Application JavaScript

// Global variables
let currentTheme = localStorage.getItem('theme') || 'light';
let toastContainer = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize theme
    initializeTheme();

    // Initialize toast container
    initializeToasts();

    // Initialize theme toggle
    initializeThemeToggle();

    // Initialize tooltips
    initializeTooltips();

    // Initialize forms
    initializeForms();

    // Initialize navigation
    initializeNavigation();

    // Initialize auto-save
    initializeAutoSave();

    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();
}

// Theme Management
function initializeTheme() {
    const html = document.documentElement;
    html.setAttribute('data-bs-theme', currentTheme);

    // Update theme icon
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.setAttribute('data-feather', currentTheme === 'dark' ? 'moon' : 'sun');
        feather.replace();
    }
}

function initializeThemeToggle() {
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', currentTheme);

    const html = document.documentElement;
    html.setAttribute('data-bs-theme', currentTheme);

    // Update icon with animation
    const themeIcon = document.getElementById('theme-icon');
    if (themeIcon) {
        themeIcon.style.transform = 'rotate(180deg)';
        setTimeout(() => {
            themeIcon.setAttribute('data-feather', currentTheme === 'dark' ? 'moon' : 'sun');
            feather.replace();
            themeIcon.style.transform = 'rotate(0deg)';
        }, 150);
    }

    // Show toast notification
    showToast(`Switched to ${currentTheme} theme`, 'info');
}

// Toast Notifications
function initializeToasts() {
    // Create toast container if it doesn't exist
    if (!document.getElementById('toast-container')) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    } else {
        toastContainer = document.getElementById('toast-container');
    }
}

function showToast(message, type = 'info', duration = 4000) {
    if (!toastContainer) {
        initializeToasts();
    }

    const toastId = 'toast-' + Date.now();
    const iconMap = {
        'success': 'check-circle',
        'error': 'alert-circle',
        'warning': 'alert-triangle',
        'info': 'info'
    };

    const colorMap = {
        'success': 'text-success',
        'error': 'text-danger',
        'warning': 'text-warning',
        'info': 'text-info'
    };

    const toast = document.createElement('div');
    toast.id = toastId;
    toast.className = 'toast align-items-center border-0 fade show';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body d-flex align-items-center">
                <i data-feather="${iconMap[type] || 'info'}" class="${colorMap[type] || 'text-info'} me-2"></i>
                <span>${message}</span>
            </div>
            <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toast);
    feather.replace();

    // Initialize Bootstrap toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: duration > 0,
        delay: duration
    });

    // Remove from DOM after hide
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });

    bsToast.show();

    return bsToast;
}

// Loading Overlay
function showLoadingOverlay(message = 'Loading...') {
    let overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('d-none');
        const messageEl = overlay.querySelector('div:last-child');
        if (messageEl) {
            messageEl.textContent = message;
        }
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('d-none');
    }
}

// Form Enhancements
function initializeForms() {
    // Add loading states to form submissions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton && !submitButton.disabled) {
                const originalText = submitButton.innerHTML;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Saving...';
                submitButton.disabled = true;

                // Re-enable after 5 seconds (fallback)
                setTimeout(() => {
                    submitButton.innerHTML = originalText;
                    submitButton.disabled = false;
                }, 5000);
            }
        });
    });

    // Auto-resize textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        autoResizeTextarea(textarea);
        textarea.addEventListener('input', () => autoResizeTextarea(textarea));
    });

    // Email validation
    const emailInputs = document.querySelectorAll('input[type="email"]');
    emailInputs.forEach(input => {
        input.addEventListener('blur', validateEmail);
    });
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 400) + 'px';
}

function validateEmail(event) {
    const email = event.target.value;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (email && !emailRegex.test(email)) {
        event.target.classList.add('is-invalid');
        showToast('Please enter a valid email address', 'error');
    } else {
        event.target.classList.remove('is-invalid');
    }
}

// Navigation Enhancements
function initializeNavigation() {
    // Highlight active navigation item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');

    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && currentPath.startsWith(href) && href !== '/') {
            link.classList.add('active');
        }
    });

    // Add smooth scrolling to anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// Tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Auto-save functionality
function initializeAutoSave() {
    let autoSaveTimer;
    const autoSaveElements = document.querySelectorAll('[data-auto-save]');

    autoSaveElements.forEach(element => {
        element.addEventListener('input', function() {
            clearTimeout(autoSaveTimer);
            autoSaveTimer = setTimeout(() => {
                autoSave(element);
            }, 2000); // Auto-save after 2 seconds of inactivity
        });
    });
}

function autoSave(element) {
    const form = element.closest('form');
    if (form && form.id === 'composeForm') {
        // Trigger save draft for compose form
        const saveDraftBtn = document.getElementById('saveDraftBtn');
        if (saveDraftBtn) {
            saveDraftBtn.click();
        }
    }
}

// Keyboard Shortcuts
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + S: Save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            const saveDraftBtn = document.getElementById('saveDraftBtn');
            if (saveDraftBtn) {
                saveDraftBtn.click();
                showToast('Draft saved (Ctrl+S)', 'success');
            }
        }

        // Ctrl/Cmd + Enter: Send email
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            const sendEmailBtn = document.getElementById('sendEmailBtn');
            if (sendEmailBtn) {
                sendEmailBtn.click();
            }
        }

        // Ctrl/Cmd + /: Show keyboard shortcuts
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            showKeyboardShortcuts();
        }

        // Escape: Close modals
        if (e.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal.show');
            openModals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
}

function showKeyboardShortcuts() {
    const shortcuts = [
        { key: 'Ctrl+S', action: 'Save draft' },
        { key: 'Ctrl+Enter', action: 'Send email' },
        { key: 'Ctrl+/', action: 'Show shortcuts' },
        { key: 'Escape', action: 'Close modal' }
    ];

    let shortcutsList = shortcuts.map(shortcut => 
        `<div class="d-flex justify-content-between py-1">
            <kbd>${shortcut.key}</kbd>
            <span>${shortcut.action}</span>
        </div>`
    ).join('');

    showToast(`
        <div class="fw-bold mb-2">Keyboard Shortcuts</div>
        ${shortcutsList}
    `, 'info', 8000);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(date) {
    if (typeof date === 'string') {
        date = new Date(date);
    }
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Animation helpers
function animateValue(element, start, end, duration) {
    const startTimestamp = performance.now();
    const step = (timestamp) => {
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    showToast('An unexpected error occurred. Please refresh the page.', 'error');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showToast('A network error occurred. Please check your connection.', 'error');
});

// Service worker registration removed to prevent console errors

// Export functions for use in other scripts
window.appUtils = {
    showToast,
    hideLoadingOverlay,
    showLoadingOverlay,
    formatDate,
    formatFileSize,
    debounce,
    throttle,
    animateValue
};

// Function to safely query the DOM
function safeQuerySelector(selector) {
        if (!selector || selector === '#') return null;
        const element = document.querySelector(selector);
        return element;
}