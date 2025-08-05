
// Email Compose Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeComposeForm();
});

function initializeComposeForm() {
    const generateBtn = document.getElementById('generateReplyBtn');
    const originalEmailInput = document.getElementById('originalEmail');
    const contextInput = document.getElementById('context');
    const toneSelect = document.getElementById('emailTone');
    const modelSelect = document.getElementById('aiModel');
    const customInstructionsInput = document.getElementById('customInstructions');
    const subjectInput = document.getElementById('emailSubject');
    const bodyInput = document.getElementById('emailBody');
    const aiResponseCard = document.getElementById('aiResponseCard');
    const aiGeneratedContent = document.getElementById('aiGeneratedContent');
    const generationTime = document.getElementById('generationTime');

    if (generateBtn) {
        generateBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            
            const originalEmail = originalEmailInput?.value || '';
            const context = contextInput?.value || '';
            const tone = toneSelect?.value || 'professional';
            const model = modelSelect?.value || 'auto';
            const customInstructions = customInstructionsInput?.value || '';

            if (!originalEmail.trim()) {
                if (window.appUtils) {
                    window.appUtils.showToast('Please enter the original email first', 'warning');
                } else {
                    alert('Please enter the original email first');
                }
                return;
            }

            // Show loading state
            generateBtn.disabled = true;
            const originalText = generateBtn.innerHTML;
            generateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
            
            showLoadingOverlay('Generating AI response...');
            
            try {
                const response = await fetch('/api/generate-reply', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        original_email: originalEmail,
                        context: context,
                        tone: tone,
                        model: model,
                        custom_instructions: customInstructions
                    })
                });

                const result = await response.json();
                console.log('AI Generation Result:', result);

                if (result.success) {
                    // Show AI response card
                    if (aiResponseCard) {
                        aiResponseCard.classList.remove('d-none');
                    }
                    
                    // Display generated content
                    if (aiGeneratedContent) {
                        let content = '';
                        if (result.subject) {
                            content += `Subject: ${result.subject}\n\n`;
                        }
                        if (result.body) {
                            content += result.body;
                        } else if (result.content) {
                            content = result.content;
                        }
                        aiGeneratedContent.textContent = content;
                    }
                    
                    // Show generation info
                    if (generationTime) {
                        const modelUsed = result.model_used || model || 'AI';
                        const timeMs = result.generation_time_ms || 1500;
                        const timeDisplay = `${(timeMs / 1000).toFixed(1)}s`;
                        generationTime.textContent = `Generated in ${timeDisplay} using ${modelUsed}`;
                    }

                    if (window.appUtils) {
                        window.appUtils.showToast('Email reply generated successfully!', 'success');
                    }
                } else {
                    const errorMsg = result.error || 'Failed to generate reply';
                    console.error('AI Generation Error:', errorMsg);
                    if (window.appUtils) {
                        window.appUtils.showToast(errorMsg, 'error');
                    } else {
                        alert('Error: ' + errorMsg);
                    }
                }
            } catch (error) {
                console.error('Error generating reply:', error);
                if (window.appUtils) {
                    window.appUtils.showToast('Network error occurred', 'error');
                } else {
                    alert('Network error occurred');
                }
            } finally {
                // Reset button
                generateBtn.disabled = false;
                generateBtn.innerHTML = originalText;
                hideLoadingOverlay();
                
                // Refresh feather icons
                if (typeof feather !== 'undefined') {
                    feather.replace();
                }
            }
        });
    }

    // Use AI Response button
    const useResponseBtn = document.getElementById('useResponseBtn');
    if (useResponseBtn) {
        useResponseBtn.addEventListener('click', function() {
            const aiContent = aiGeneratedContent?.textContent || '';
            if (!aiContent) {
                if (window.appUtils) {
                    window.appUtils.showToast('No AI content to use', 'warning');
                } else {
                    alert('No AI content to use');
                }
                return;
            }

            const lines = aiContent.split('\n');
            let subject = '';
            let body = '';
            let isBody = false;

            lines.forEach(line => {
                if (line.startsWith('Subject:')) {
                    subject = line.replace('Subject:', '').trim();
                } else if (isBody || (!line.startsWith('Subject:') && line.trim() !== '')) {
                    if (!line.startsWith('Subject:')) {
                        isBody = true;
                        body += line + '\n';
                    }
                }
            });

            if (subject && subjectInput) {
                subjectInput.value = subject;
            }
            if (body && bodyInput) {
                bodyInput.value = body.trim();
                // Auto-resize if it's a textarea
                if (bodyInput.tagName === 'TEXTAREA') {
                    if (window.appUtils) {
                        window.appUtils.autoResizeTextarea(bodyInput);
                    }
                }
            }

            if (aiResponseCard) {
                aiResponseCard.classList.add('d-none');
            }

            if (window.appUtils) {
                window.appUtils.showToast('AI response applied to email', 'success');
            }
        });
    }
}

// Export for global use
window.composeUtils = {
    initializeComposeForm
};
