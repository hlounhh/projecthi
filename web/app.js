(function() {
  var videos = document.querySelectorAll('.fr-auto-video');
  if (!videos.length) return;

  function tryPlay(v) {
    var p = v.play();
    if (p && typeof p.then === 'function') {
      p.catch(function(e) {});
    }
  }

  if ('IntersectionObserver' in window) {
    var videoObserver = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        var vid = entry.target;
        if (entry.isIntersecting) {
          tryPlay(vid);
        } else {
          vid.pause();
        }
      });
    }, { threshold: 0.15 });

    videos.forEach(function(v) {
      videoObserver.observe(v);
    });
  } else {
    videos.forEach(function(v) {
      tryPlay(v);
    });
  }
})();
