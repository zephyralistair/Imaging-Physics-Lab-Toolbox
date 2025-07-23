import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def ortho_viewer(
        vol,                            # overlay (Z, Y, X)
        underlay=None,                  # background (optional)
        alpha=0.6,                      # overlay opacity
        vmax=None,                      # overlay vmax (default: max of vol)
        spacing=(1., 1., 1.),           # (sz, sy, sx) mm
        init_xyz=None,                  # start voxel
        cmap='viridis',
        underlay_cmap='gray',
        figsize=(9, 6)):
    """
    Axial (TL) · Sagittal (rot 90°) (TR) · Coronal (BL) viewer.

    • Axial & coronal **same width** (X dimension)  
    • Axial & sagittal **same height** (Y dimension)  
    • Three sliders stacked in bottom-right corner.
    """
    # ── sanity ────────────────────────────────────────────────────────────
    if vol.ndim != 3:
        raise ValueError("vol must be 3-D (Z, Y, X)")
    if underlay is not None and underlay.shape != vol.shape:
        raise ValueError("underlay must match vol shape")

    zmax, ymax, xmax = vol.shape
    sz,  sy,   sx    = spacing
    if init_xyz is None:
        init_xyz = (zmax//2, ymax//2, xmax//2)
    idx = list(map(int, init_xyz))

    # ── physical extents dictate the GridSpec ratios ─────────────────────
    w_axial   = xmax * sx               # width  of axial  (and coronal)
    w_sagittal= zmax * sz               # width  of sagittal
    h_axial   = ymax * sy               # height of axial (and sagittal)
    h_coronal = zmax * sz               # height of coronal
    h_slider  = 0.10 * h_axial
    min_slider_px = 25
    dpi           = plt.rcParams['figure.dpi']
    h_slider      = max(h_slider, min_slider_px / dpi)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                    wspace=0.02, hspace=0.02)
    # plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.02)
    gs  = fig.add_gridspec(
            3, 2,
            width_ratios =[w_axial, w_sagittal],
            height_ratios=[h_axial, h_coronal, h_slider])

    ax_axial    = fig.add_subplot(gs[0, 0])
    ax_sagittal = fig.add_subplot(gs[0, 1])
    ax_coronal  = fig.add_subplot(gs[1, 0])

    # empty cell bottom-right reserved for sliders
    fig.add_subplot(gs[1, 1]).set_axis_off()
    spacer = fig.add_subplot(gs[2, 0])
    spacer.set_axis_off()              # blank strip under coronal

    for ax in (ax_axial, ax_sagittal, ax_coronal):
        ax.set_axis_off()

    # ── draw helper (underlay → overlay) ─────────────────────────────────
    def _two_layer(ax, data, bg=None, rotate=False):
        if rotate:
            data = data.T
            bg   = None if bg is None else bg.T
        im_bg = ax.imshow(bg, cmap=underlay_cmap,
                          origin='lower') if bg is not None else None
        im_fg = ax.imshow(data, cmap=cmap, alpha=alpha, origin='lower', 
                          vmin=0, vmax=vmax) if vmax is not None else \
                ax.imshow(data, cmap=cmap, alpha=alpha, origin='lower')
        return im_bg, im_fg

    bg_ax, fg_ax = _two_layer(
        ax_axial, vol[idx[0], :, :],
        None if underlay is None else underlay[idx[0], :, :])

    bg_sag, fg_sag = _two_layer(
        ax_sagittal, vol[:, :, idx[2]],
        None if underlay is None else underlay[:, :, idx[2]], rotate=True)

    bg_cor, fg_cor = _two_layer(
        ax_coronal, vol[:, idx[1], :],
        None if underlay is None else underlay[:, idx[1], :])

    # ------------------------------------------------------------
    # build and draw once so constrained_layout finalises positions
    fig.canvas.draw()

    # Left-edge & width of the sagittal axes (figure-fraction coords)
    sag_box   = ax_sagittal.get_position()
    x0        = sag_box.x0
    slider_w  = sag_box.width

    # vertical layout
    slider_h  = 0.03
    gap       = 0.01
    y0        = 0.06                     # bottom of lowest slider stays the same

    ax_sz = fig.add_axes([x0, y0 + 2*(slider_h + gap), slider_w, slider_h])
    ax_sy = fig.add_axes([x0, y0 +    slider_h + gap , slider_w, slider_h])
    ax_sx = fig.add_axes([x0, y0,                      slider_w, slider_h])

    s_z = Slider(ax_sz, 'Axial',    0, zmax-1, valinit=idx[0], valstep=1)   # ③ NEW names
    s_y = Slider(ax_sy, 'Coronal',  0, ymax-1, valinit=idx[1], valstep=1)
    s_x = Slider(ax_sx, 'Sagittal', 0, xmax-1, valinit=idx[2], valstep=1)
    sliders = (s_z, s_y, s_x)

    # ── titles & cross-hairs ─────────────────────────────────────────────
    def _titles():
        z, y, x = idx
        return (f'Axial   z={z} ({z*sz:.1f} mm)',
                f'Sagittal x={x} ({x*sx:.1f} mm)',
                f'Coronal y={y} ({y*sy:.1f} mm)')

    ax_axial.set_title(_titles()[0])
    ax_sagittal.set_title(_titles()[1])
    ax_coronal.set_title(_titles()[2])

    v_ax = ax_axial.axvline(0, ls='--', c='yellow', lw=0.8)
    h_ax = ax_axial.axhline(0, ls='--', c='yellow', lw=0.8)

    v_sag = ax_sagittal.axvline(0, ls='--', c='yellow', lw=0.8)
    h_sag = ax_sagittal.axhline(0, ls='--', c='yellow', lw=0.8)

    v_cor = ax_coronal.axvline(0, ls='--', c='yellow', lw=0.8)
    h_cor = ax_coronal.axhline(0, ls='--', c='yellow', lw=0.8)

    # correct mapping after 90° rot of sagittal --------------------------
    def _update_cross():
        v_ax.set_xdata([idx[2]]); h_ax.set_ydata([idx[1]])
        v_sag.set_xdata([idx[0]]); h_sag.set_ydata([idx[1]])   # <-- fixed
        v_cor.set_xdata([idx[2]]); h_cor.set_ydata([idx[0]])

    # ── redraw function ─────────────────────────────────────────────────
    def _refresh():
        fg_ax.set_data(vol[idx[0], :, :])
        if underlay is not None:
            bg_ax.set_data(underlay[idx[0], :, :])

        fg_sag.set_data(vol[:, :, idx[2]].T)
        if underlay is not None:
            bg_sag.set_data(underlay[:, :, idx[2]].T)

        fg_cor.set_data(vol[:, idx[1], :])
        if underlay is not None:
            bg_cor.set_data(underlay[:, idx[1], :])

        t_ax, t_sag, t_cor = _titles()
        ax_axial.set_title(t_ax); ax_sagittal.set_title(t_sag); ax_coronal.set_title(t_cor)
        _update_cross()
        fig.canvas.draw_idle(); fig.canvas.flush_events()

    # ── slider callback ────────────────────────────────────────────────
    def _slider(_=None):
        idx[0] = int(s_z.val); idx[1] = int(s_y.val); idx[2] = int(s_x.val)
        _refresh()

    for s in sliders:
        s.on_changed(_slider)

    # ── mouse click handler ────────────────────────────────────────────
    pane_map = {ax_axial: 0, ax_sagittal: 1, ax_coronal: 2}

    def _click(event):
        ax = event.inaxes
        if ax not in pane_map or event.xdata is None or event.ydata is None:
            return
        pane = pane_map[ax]
        i, j = int(round(event.ydata)), int(round(event.xdata))

        if pane == 0:             # axial
            idx[1], idx[2] = i, j
        elif pane == 1:           # sagittal (rotated)
            idx[0], idx[1] = j, i   # swap (z ← xdata, y ← ydata)
        elif pane == 2:           # coronal
            idx[0], idx[2] = i, j

        idx[0] = np.clip(idx[0], 0, zmax-1)
        idx[1] = np.clip(idx[1], 0, ymax-1)
        idx[2] = np.clip(idx[2], 0, xmax-1)

        for s in sliders: s.eventson = False
        s_z.set_val(idx[0]); s_y.set_val(idx[1]); s_x.set_val(idx[2])
        for s in sliders: s.eventson = True
        _refresh()

    fig.canvas.mpl_connect('button_press_event', _click)

    # hide ipympl "Figure n" header (ignored outside ipympl)
    try: fig.canvas.header_visible = False
    except AttributeError: pass

    _update_cross(); _refresh()
    return {'fig': fig, 'axes': (ax_axial, ax_sagittal, ax_coronal),
            'sliders': sliders, '_idx': idx, '_refresh': _refresh}