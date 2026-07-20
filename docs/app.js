document.addEventListener('DOMContentLoaded', () => {
    // -----------------------------------------------------------------
    // Mutual Exclusion Playback Logic
    // -----------------------------------------------------------------
    const audioElements = document.querySelectorAll('audio');
    let currentPlayingAudio = null;

    audioElements.forEach(audio => {
        const card = audio.closest('.audio-card') || audio.closest('.journey-item');

        audio.addEventListener('play', () => {
            // Pause any currently playing track
            if (currentPlayingAudio && currentPlayingAudio !== audio) {
                currentPlayingAudio.pause();
                const playingCard = currentPlayingAudio.closest('.audio-card') || currentPlayingAudio.closest('.journey-item');
                if (playingCard) {
                    playingCard.classList.remove('active-playing');
                }
            }

            currentPlayingAudio = audio;
            if (card) {
                card.classList.add('active-playing');
            }
        });

        audio.addEventListener('pause', () => {
            if (currentPlayingAudio === audio && card) {
                card.classList.remove('active-playing');
            }
        });

        audio.addEventListener('ended', () => {
            if (card) {
                card.classList.remove('active-playing');
            }
            if (currentPlayingAudio === audio) {
                currentPlayingAudio = null;
            }
        });
    });
});
