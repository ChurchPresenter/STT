// fetch() for endpoints a page polls on a timer: stop hammering one that
// keeps refusing.
//
// A page left open on a machine that is not in the settings whitelist
// polls forever and is refused every time — one display client produced
// 9,182 consecutive 403s in a week, about a fifth of the whole access
// log, and every one of them displaced a real request inside the log's
// row cap. The same shape appears with 404 when the two paired machines
// are on different versions: the newer one polls an endpoint the older
// one does not have yet, for as long as the version gap lasts.
//
// So after a few consecutive 403/404s the endpoint is left alone for a
// while. The caller's timer keeps ticking — it just returns a rejected
// promise without touching the network, so no page has to restructure
// its polling loop. A single success clears the state, and the pause
// expires on its own, so whitelisting the machine or finishing the
// update recovers without anyone reloading the page.
window.pollFetch = (function () {
    var LIMIT = 3;              // consecutive refusals before pausing
    var PAUSE_MS = 10 * 60000;  // how long to leave the endpoint alone
    var state = {};             // path -> {denials, pausedUntil}

    // Clear a pause. A button the operator just pressed is not polling —
    // it must try the server again however the timer behind it is doing.
    function reset(url) {
        delete state[String(url).split('?')[0]];
    }

    var poll = function (url, options) {
        var key = String(url).split('?')[0];
        var entry = state[key] || (state[key] = { denials: 0, pausedUntil: 0 });
        var now = Date.now();
        if (entry.pausedUntil > now) {
            var err = new Error('polling paused for ' + key);
            err.pollingPaused = true;
            return Promise.reject(err);
        }
        return fetch(url, options).then(function (response) {
            if (response.status === 403 || response.status === 404) {
                entry.denials += 1;
                if (entry.denials >= LIMIT) {
                    entry.pausedUntil = Date.now() + PAUSE_MS;
                    console.warn('pollFetch: ' + key + ' returned ' +
                        response.status + ' ' + entry.denials +
                        ' times — pausing polling for ' +
                        (PAUSE_MS / 60000) + ' minutes');
                }
            } else {
                entry.denials = 0;
                entry.pausedUntil = 0;
            }
            return response;
        });
        // A rejected fetch (server restarting, network blip) is deliberately
        // not counted: that is the case where polling SHOULD keep trying.
    };
    poll.reset = reset;
    return poll;
})();
