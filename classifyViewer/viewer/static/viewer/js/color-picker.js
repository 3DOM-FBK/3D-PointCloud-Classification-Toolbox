// Colour picker popover used for the viewport background.
// Saturation/value area, hue slider, hex field and a few presets; styled with the
// viewer's CSS variables (see the .color-picker rules in styles.css).

const PRESET_COLORS = ['#121214', '#1e1e20', '#2d2d30', '#1e3a5f', '#000000', '#ffffff', '#f5f5f7', '#d1d1d6'];

let openPicker = null;

export function hexToRgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgbToHex({ r, g, b }) {
    return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}

function rgbToHsv({ r, g, b }) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
        if (max === r) h = ((g - b) / d) % 6;
        else if (max === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
        if (h < 0) h += 360;
    }
    return { h, s: max === 0 ? 0 : d / max, v: max };
}

function hsvToRgb({ h, s, v }) {
    const c = v * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = v - c;
    const [r, g, b] = h < 60 ? [c, x, 0]
        : h < 120 ? [x, c, 0]
        : h < 180 ? [0, c, x]
        : h < 240 ? [0, x, c]
        : h < 300 ? [x, 0, c]
        : [c, 0, x];
    return { r: (r + m) * 255, g: (g + m) * 255, b: (b + m) * 255 };
}

const clamp01 = (v) => Math.min(Math.max(v, 0), 1);

/**
 * Opens the picker under `anchor`, starting from `initialHex`.
 * `onChange(hex)` is called with a lowercase #rrggbb value at every change.
 * Clicking the same anchor again, clicking outside or pressing Esc closes it.
 */
export function openColorPicker(anchor, initialHex, onChange) {
    if (openPicker) {
        const sameAnchor = openPicker.anchor === anchor;
        openPicker.close();
        if (sameAnchor) return null;
    }

    let { h, s, v } = rgbToHsv(hexToRgb(initialHex));

    const el = document.createElement('div');
    el.className = 'color-picker';
    el.innerHTML = `
        <div class="color-picker-sv"><div class="color-picker-thumb"></div></div>
        <input type="range" class="color-picker-hue" min="0" max="359" step="1" aria-label="Hue">
        <div class="color-picker-footer">
            <span class="color-picker-preview"></span>
            <input type="text" class="property-input color-picker-hex" maxlength="7" spellcheck="false" aria-label="Hex colour">
        </div>
        <div class="color-picker-presets"></div>`;

    const sv = el.querySelector('.color-picker-sv');
    const thumb = el.querySelector('.color-picker-thumb');
    const hueInput = el.querySelector('.color-picker-hue');
    const preview = el.querySelector('.color-picker-preview');
    const hexInput = el.querySelector('.color-picker-hex');
    const presetsEl = el.querySelector('.color-picker-presets');

    const currentHex = () => rgbToHex(hsvToRgb({ h, s, v }));

    // Redraws the controls from h/s/v. Does not notify the caller.
    function render() {
        sv.style.background =
            'linear-gradient(to top, #000, rgba(0, 0, 0, 0)), ' +
            `linear-gradient(to right, #fff, hsl(${h}, 100%, 50%))`;
        thumb.style.left = `${s * 100}%`;
        thumb.style.top = `${(1 - v) * 100}%`;
        hueInput.value = Math.round(h);
        preview.style.backgroundColor = currentHex();
        if (document.activeElement !== hexInput) hexInput.value = currentHex();
    }

    function update() {
        render();
        onChange(currentHex());
    }

    function setFromHex(hex) {
        ({ h, s, v } = rgbToHsv(hexToRgb(hex)));
    }

    // Saturation/value area: drag anywhere inside it
    const moveInSv = (e) => {
        const r = sv.getBoundingClientRect();
        s = clamp01((e.clientX - r.left) / r.width);
        v = clamp01(1 - (e.clientY - r.top) / r.height);
        update();
    };
    sv.addEventListener('pointerdown', (e) => {
        sv.setPointerCapture(e.pointerId);
        moveInSv(e);
    });
    sv.addEventListener('pointermove', (e) => {
        if (sv.hasPointerCapture(e.pointerId)) moveInSv(e);
    });

    hueInput.addEventListener('input', () => {
        h = Number(hueInput.value);
        update();
    });

    // Enter / blur commits the typed value; an invalid value is reverted
    hexInput.addEventListener('change', () => {
        let text = hexInput.value.trim();
        if (!text.startsWith('#')) text = '#' + text;
        if (/^#[0-9a-f]{6}$/i.test(text)) {
            setFromHex(text.toLowerCase());
            update();
        } else {
            render();
        }
    });

    for (const color of PRESET_COLORS) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'color-picker-preset';
        btn.style.backgroundColor = color;
        btn.title = color;
        btn.addEventListener('click', () => {
            setFromHex(color);
            update();
        });
        presetsEl.appendChild(btn);
    }

    document.body.appendChild(el);

    function position() {
        const r = anchor.getBoundingClientRect();
        const w = el.offsetWidth;
        const height = el.offsetHeight;
        let top = r.bottom + 8;
        if (top + height > window.innerHeight - 8) top = Math.max(8, r.top - height - 8);
        const left = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
        el.style.top = `${top}px`;
        el.style.left = `${left}px`;
    }

    function onPointerDownOutside(e) {
        if (el.contains(e.target) || anchor.contains(e.target)) return;
        api.close();
    }
    function onKeyDown(e) {
        if (e.key === 'Escape') api.close();
    }

    const api = {
        anchor,
        close() {
            document.removeEventListener('pointerdown', onPointerDownOutside, true);
            document.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('resize', api.close);
            el.remove();
            if (openPicker === api) openPicker = null;
        },
    };

    render();
    position();
    document.addEventListener('pointerdown', onPointerDownOutside, true);
    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('resize', api.close);
    openPicker = api;
    return api;
}
