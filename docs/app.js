document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------------------------------
    // Tab Navigation Logic
    // -----------------------------------------------------------------
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active classes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanels.forEach(panel => panel.classList.remove('active'));

            // Add active class to clicked button
            button.classList.add('active');
            
            // Show corresponding panel
            const tabId = button.getAttribute('data-tab');
            const targetPanel = document.getElementById(`tab-${tabId}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });

    // -----------------------------------------------------------------
    // Audio Playback & Web Audio API Visualizer Setup
    // -----------------------------------------------------------------
    const audioElements = document.querySelectorAll('audio');
    const visualizerContainer = document.getElementById('visualizer-container');
    const activeTrackTitle = document.getElementById('active-track-title');
    const activeTrackBadge = document.getElementById('active-track-type');
    const speedSelect = document.getElementById('playback-speed');
    const muteBtn = document.getElementById('btn-mute');
    
    const canvas = document.getElementById('audio-visualizer-canvas');
    const canvasCtx = canvas.getContext('2d');

    let audioCtx = null;
    let analyser = null;
    let currentPlayingAudio = null;
    const sourceNodes = new Map(); // Track audio elements to avoid double-binding source nodes

    // Resize canvas to match its client width
    function resizeCanvas() {
        canvas.width = canvas.parentElement.clientWidth;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Initialize Web Audio Context
    function initWebAudio(audioElement) {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 512;
            analyser.connect(audioCtx.destination);
            
            // Start visualizer draw loop
            drawVisualizer();
        }

        // Only create source node once per element
        if (!sourceNodes.has(audioElement)) {
            try {
                const source = audioCtx.createMediaElementSource(audioElement);
                source.connect(analyser);
                sourceNodes.set(audioElement, source);
            } catch (err) {
                console.warn("Could not bind audio source to Web Audio: ", err);
            }
        }
    }

    // Mutual exclusion audio playback
    audioElements.forEach(audio => {
        // Find parent card details
        const card = audio.closest('.audio-card');
        const trackName = card.getAttribute('data-track-name') || 'Audio Track';
        const trackType = card.getAttribute('data-track-type') || 'Audio';

        audio.addEventListener('play', () => {
            // Pause any currently playing track
            if (currentPlayingAudio && currentPlayingAudio !== audio) {
                currentPlayingAudio.pause();
                currentPlayingAudio.closest('.audio-card').classList.remove('active-playing');
            }

            currentPlayingAudio = audio;
            card.classList.add('active-playing');
            
            // Initialize visualizer context and make visualizer panel visible
            initWebAudio(audio);
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume();
            }

            visualizerContainer.style.display = 'block';
            activeTrackTitle.textContent = trackName;
            activeTrackBadge.textContent = trackType;
            
            // Reset controls to match current state
            audio.playbackRate = parseFloat(speedSelect.value);
            updateMuteButtonState(audio.muted);
            
            resizeCanvas();
        });

        audio.addEventListener('pause', () => {
            if (currentPlayingAudio === audio) {
                card.classList.remove('active-playing');
            }
        });

        audio.addEventListener('ended', () => {
            card.classList.remove('active-playing');
            if (currentPlayingAudio === audio) {
                currentPlayingAudio = null;
            }
        });
    });

    // Playback Speed Controller
    speedSelect.addEventListener('change', (e) => {
        const rate = parseFloat(e.target.value);
        if (currentPlayingAudio) {
            currentPlayingAudio.playbackRate = rate;
        }
    });

    // Mute/Unmute Controller
    muteBtn.addEventListener('click', () => {
        if (currentPlayingAudio) {
            currentPlayingAudio.muted = !currentPlayingAudio.muted;
            updateMuteButtonState(currentPlayingAudio.muted);
        }
    });

    function updateMuteButtonState(isMuted) {
        if (isMuted) {
            muteBtn.innerHTML = `<svg viewBox="0 0 24 24" class="icon"><path d="M3.63 3.63L2.36 4.9 7.46 10H3v4h4l5 5v-6.54l4.9 4.9c-.83.63-1.77 1.09-2.9 1.28v2.02c1.69-.25 3.2-.97 4.41-1.99l2.78 2.78 1.27-1.27L3.63 3.63zM12 4L9.91 6.09 12 8.18V4z"/></svg>`;
            muteBtn.title = "Unmute";
        } else {
            muteBtn.innerHTML = `<svg viewBox="0 0 24 24" class="icon"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>`;
            muteBtn.title = "Mute";
        }
    }

    // -----------------------------------------------------------------
    // Real-time Oscilloscope Renderer (Web Audio API)
    // -----------------------------------------------------------------
    function drawVisualizer() {
        if (!analyser) return;

        requestAnimationFrame(drawVisualizer);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        // Clear canvas
        canvasCtx.fillStyle = '#ffffff';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw central reference grid line
        canvasCtx.lineWidth = 1;
        canvasCtx.strokeStyle = '#e2e8f0';
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, canvas.height / 2);
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();

        // Draw active waveform
        canvasCtx.lineWidth = 1.5;
        canvasCtx.strokeStyle = '#2b6cb0'; // Academic Blue
        canvasCtx.beginPath();

        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }
});
