// --- Mobile Navigation ---
const navToggle = document.getElementById('navToggle');
const mobileNav = document.getElementById('mobileNav');

if (navToggle && mobileNav) {
  navToggle.addEventListener('click', () => {
    const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', !isExpanded);
    mobileNav.classList.toggle('is-open');
  });
}

// --- Theme Toggler ---
const themeBtn = document.querySelector('button[title="Toggle Dark Theme"]');

function updateThemeIcon(theme) {
  if (themeBtn) {
    themeBtn.textContent = theme === 'dark' ? '☀️' : '🌙';
  }
}

function initTheme() {
  const storedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = storedTheme || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
  updateThemeIcon(theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const target = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', target);
  localStorage.setItem('theme', target);
  updateThemeIcon(target);
}

// Expose to HTML inline onclick handlers
window.toggleTheme = toggleTheme;
initTheme();

// Give keyboard users a consistent way to bypass repeated navigation.
if (document.querySelector("main") && !document.querySelector(".skip-link")) {
  const skip = document.createElement("a");
  skip.className = "skip-link";
  skip.href = "#main-content";
  skip.textContent = "Skip to main content";
  document.body.prepend(skip);
  const main = document.querySelector("main");
  if (!main.id) main.id = "main-content";
}

// --- Smooth Page Transitions ---
document.addEventListener('click', (e) => {
  const link = e.target.closest('a');
  
  // Only intercept local links (ignore external links, target="_blank", or #hash jumps)
  if (link && link.host === window.location.host && !link.hash && link.target !== '_blank') {
    e.preventDefault(); // Stop the hard flash
    document.body.classList.add('fade-out'); // Trigger CSS fade out
    
    setTimeout(() => {
      window.location.href = link.href; // Navigate after fade out
    }, 250); // Matches the CSS transition time
  }
});

function showToast(message, type = "") {
  let stack = document.querySelector(".toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.className = "toast-stack";
    stack.setAttribute("aria-live", "polite");
    document.body.appendChild(stack);
  }
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "is-error" : ""}`;
  toast.textContent = message;
  stack.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}