// Advanced Hybrid RAG: sign-in lives in ui_service (/login), not in Chainlit.
// When Chainlit shows its own login screen (session expired or logged out),
// send the browser to the real login page and come back to the chat after.
(function () {
  function redirectIfLogin() {
    var p = window.location.pathname;
    if (p === "/chat/login" || p.indexOf("/chat/login/") === 0) {
      window.location.replace("/login?next=/chat");
    }
  }
  redirectIfLogin();
  var push = history.pushState;
  history.pushState = function () {
    var r = push.apply(this, arguments);
    redirectIfLogin();
    return r;
  };
  var replace = history.replaceState;
  history.replaceState = function () {
    var r = replace.apply(this, arguments);
    redirectIfLogin();
    return r;
  };
  window.addEventListener("popstate", redirectIfLogin);
})();
