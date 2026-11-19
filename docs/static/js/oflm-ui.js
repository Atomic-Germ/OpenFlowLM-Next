/* OpenFlowLM lightweight floating tooltip for [data-oflm-tip] badges (NPU + CPU breakdown, etc.) */
(function () {
    'use strict';

    let tipEl = null;
    let activeTarget = null;

    function ensureTip() {
        if (tipEl) return tipEl;
        tipEl = document.createElement('div');
        tipEl.className = 'oflm-floating-tip';
        tipEl.setAttribute('role', 'tooltip');
        tipEl.setAttribute('aria-hidden', 'true');
        document.body.appendChild(tipEl);
        return tipEl;
    }

    function positionTip(target) {
        if (!tipEl || !target) return;
        const rect = target.getBoundingClientRect();
        const pad = 10;
        tipEl.style.left = '0px';
        tipEl.style.top = '0px';
        const tipRect = tipEl.getBoundingClientRect();

        let left = rect.left + (rect.width / 2) - (tipRect.width / 2);
        left = Math.max(pad, Math.min(window.innerWidth - tipRect.width - pad, left));

        let top = rect.top - tipRect.height - 8;
        if (top < pad) {
            top = rect.bottom + 8;
        }
        tipEl.style.left = Math.round(left) + 'px';
        tipEl.style.top = Math.round(top) + 'px';
    }

    function showTip(target) {
        const text = target && target.getAttribute('data-oflm-tip');
        if (!text) return;
        activeTarget = target;
        const el = ensureTip();
        el.textContent = text;
        el.classList.add('is-visible');
        el.setAttribute('aria-hidden', 'false');
        positionTip(target);
    }

    function hideTip() {
        activeTarget = null;
        if (!tipEl) return;
        tipEl.classList.remove('is-visible');
        tipEl.setAttribute('aria-hidden', 'true');
    }

    document.addEventListener('mouseover', function (e) {
        const t = e.target && e.target.closest ? e.target.closest('[data-oflm-tip]') : null;
        if (t && t !== activeTarget) {
            showTip(t);
        } else if (!t && activeTarget) {
            hideTip();
        }
    });

    document.addEventListener('focusin', function (e) {
        const t = e.target && e.target.closest ? e.target.closest('[data-oflm-tip]') : null;
        if (t) showTip(t);
    });

    document.addEventListener('focusout', function (e) {
        const t = e.target && e.target.closest ? e.target.closest('[data-oflm-tip]') : null;
        if (t && t === activeTarget) hideTip();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && activeTarget) hideTip();
    });

    window.addEventListener('scroll', function () {
        if (activeTarget) hideTip();
    }, { passive: true });

    window.addEventListener('resize', function () {
        if (activeTarget) positionTip(activeTarget);
    }, { passive: true });
})();
