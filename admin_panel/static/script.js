// For Dashboard: Real-time Parked Count
document.addEventListener('DOMContentLoaded', function() {
    const liveParkedCountElement = document.getElementById('live-parked-count');

    if (liveParkedCountElement) {
        function updateParkedCount() {
            fetch('/api/parked_count')
                .then(response => response.json())
                .then(data => {
                    liveParkedCountElement.innerText = data.parked_count;
                })
                .catch(error => console.error('Error fetching parked count:', error));
        }

        // Update every 5 seconds
        setInterval(updateParkedCount, 5000);
        // Initial update when page loads
        updateParkedCount();
    }

    // For Live Camera Feed Page
    const liveCameraFeedElement = document.getElementById('live-camera-feed');
    if (liveCameraFeedElement) {
        function refreshLiveFeed() {
            // Append a timestamp to the URL to force the browser to reload the image, not use cache
            liveCameraFeedElement.src = liveCameraFeedElement.src.split('?')[0] + "?t=" + new Date().getTime();
        }

        // Refresh every 100 milliseconds (10 frames per second)
        setInterval(refreshLiveFeed, 100);
    }
});