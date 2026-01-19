class UI:

    def __init__(self, mode: str = 'dark', dpi=150):
        """
        Set dark mode styling for matplotlib plots and Jupyter widgets.

        Example:
            from insardev.UI import UI
            UI('dark')
        """
        import matplotlib.pyplot as plt
        from IPython.display import HTML, display

        if mode not in ['dark', 'light']:
            raise ValueError("Invalid mode. Must be 'dark' or 'light'.")

        plt.rcParams['figure.figsize'] = [12, 4]
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        if mode != 'dark':
            return

        # Matplotlib dark theme
        plt.rcParams.update({
            'figure.facecolor': 'black',
            'axes.facecolor': 'black',
            'savefig.facecolor': 'black',
            'text.color': 'lightgray',
            'axes.labelcolor': 'lightgray',
            'xtick.color': 'lightgray',
            'ytick.color': 'lightgray',
            'axes.edgecolor': 'lightgray'
        })

        # CSS for ipywidgets (excluding leaflet/pyvista)
        dark_css = """
        <style>
        .widget-box,
        .widget-text, .widget-int-text, .widget-float-text,
        .widget-dropdown,
        .jp-InputPrompt,
        .cell-output-ipywidget-background,
        .cell-output-ipywidget-background *:not([class*="leaflet"]):not([class*="pv-"]):not(canvas):not(svg):not(svg *) {
            background-color: #333333 !important;
            color: lightgray !important;
            border-color: #555555 !important;
            outline: none !important;
        }

        .leaflet-container,
        .leaflet-container *,
        [class*="leaflet-"],
        [class*="pv-"] {
            background-color: unset !important;
            color: unset !important;
            border-color: unset !important;
        }

        .widget-label {
            color: white !important;
        }

        .widget-button {
            background-color: #444444 !important;
            color: lightgray !important;
            border-color: #555555 !important;
        }

        .widget-html-content {
            color: lightgray !important;
        }

        .jupyter-widgets .widget-html-content,
        .jupyter-widget-html-content {
            font-family: monospace !important;
        }
        </style>
        """
        display(HTML(dark_css))

        # Monkey-patch xarray for dark theme HTML repr
        try:
            import xarray as xr

            xarray_dark_css = """
            <style>
            .xr-wrap, .xr-wrap * { background-color: #2b2b2b !important; color: #d0d0d0 !important; border-color: #444 !important; box-shadow: none !important; }
            .xr-header { background-color: #1f1f1f !important; color: #e0e0e0 !important; }
            .xr-section-summary, .xr-section-inline-details { background-color: #2b2b2b !important; color: #d0d0d0 !important; }
            .xr-var-list, .xr-var-item { background-color: #2b2b2b !important; color: #d0d0d0 !important; }
            .xr-var-name, .xr-var-dims, .xr-var-dtype, .xr-var-preview { color: #d0d0d0 !important; }
            .xr-index-name { color: #d0d0d0 !important; }
            .xr-array-wrap { background-color: #252525 !important; }
            pre.xr-text-repr-fallback { background-color: #252525 !important; color: #cfcfcf !important; }
            </style>
            """

            if hasattr(xr.DataArray, '_repr_html_') and not getattr(xr.DataArray._repr_html_, '_dark_patched', False):
                _original_da_repr = xr.DataArray._repr_html_
                def _dark_da_repr(self):
                    html = _original_da_repr(self)
                    return xarray_dark_css + html if html else html
                _dark_da_repr._dark_patched = True
                xr.DataArray._repr_html_ = _dark_da_repr

            if hasattr(xr.Dataset, '_repr_html_') and not getattr(xr.Dataset._repr_html_, '_dark_patched', False):
                _original_ds_repr = xr.Dataset._repr_html_
                def _dark_ds_repr(self):
                    html = _original_ds_repr(self)
                    return xarray_dark_css + html if html else html
                _dark_ds_repr._dark_patched = True
                xr.Dataset._repr_html_ = _dark_ds_repr
        except ImportError:
            pass

        # Monkey-patch ipywidgets for dark theme (handles tqdm progress bars)
        try:
            import ipywidgets as widgets

            if not getattr(widgets.HBox, '_dark_patched', False):
                _original_hbox_init = widgets.HBox.__init__
                def _dark_hbox_init(self, *args, **kwargs):
                    _original_hbox_init(self, *args, **kwargs)
                    self.layout.background = '#333333'
                widgets.HBox.__init__ = _dark_hbox_init
                widgets.HBox._dark_patched = True

            if not getattr(widgets.VBox, '_dark_patched', False):
                _original_vbox_init = widgets.VBox.__init__
                def _dark_vbox_init(self, *args, **kwargs):
                    _original_vbox_init(self, *args, **kwargs)
                    self.layout.background = '#333333'
                widgets.VBox.__init__ = _dark_vbox_init
                widgets.VBox._dark_patched = True

            if not getattr(widgets.FloatProgress, '_dark_patched', False):
                _original_progress_init = widgets.FloatProgress.__init__
                def _dark_progress_init(self, *args, **kwargs):
                    _original_progress_init(self, *args, **kwargs)
                    self.style.bar_color = '#4caf50'
                    self.layout.background = '#555555'
                widgets.FloatProgress.__init__ = _dark_progress_init
                widgets.FloatProgress._dark_patched = True

            if not getattr(widgets.HTML, '_dark_patched', False):
                _original_html_init = widgets.HTML.__init__
                def _dark_html_init(self, *args, **kwargs):
                    _original_html_init(self, *args, **kwargs)
                    self.layout.background = '#333333'
                widgets.HTML.__init__ = _dark_html_init
                widgets.HTML._dark_patched = True

        except ImportError:
            pass

        # Monkey-patch PyVista trame Widget for dark theme (removes light gray border)
        try:
            from pyvista.trame.jupyter import Widget as PyVistaWidget
            from ipywidgets import HTML

            if not getattr(PyVistaWidget, '_dark_patched', False):
                _original_pv_widget_init = PyVistaWidget.__init__
                def _dark_pv_widget_init(self, viewer, src, width=None, height=None, iframe_attrs=None, **kwargs):
                    if iframe_attrs is None:
                        iframe_attrs = {}
                    # Override border to black/none for dark theme
                    border = 'border: none; background-color: black;'
                    iframe_attrs = {
                        **iframe_attrs,
                        'src': src,
                        'class': 'pyvista',
                        'style': f'width: {width}; height: {height}; {border}',
                    }
                    iframe_attrs_str = ' '.join(f'{key}="{value!s}"' for key, value in iframe_attrs.items())
                    value = f'<iframe {iframe_attrs_str}></iframe>'
                    HTML.__init__(self, value, **kwargs)
                    self._viewer = viewer
                    self._src = src
                PyVistaWidget.__init__ = _dark_pv_widget_init
                PyVistaWidget._dark_patched = True
        except ImportError:
            pass

        # Monkey-patch PyVista trame Viewer.ui for dark theme (inject CSS into trame app)
        try:
            from pyvista.trame.ui.vuetify3 import Viewer as PyVistaViewer
            from trame.widgets import html as trame_html

            if not getattr(PyVistaViewer, '_dark_ui_patched', False):
                _original_ui = PyVistaViewer.ui
                def _dark_ui(self, *args, **kwargs):
                    result = _original_ui(self, *args, **kwargs)
                    # Inject dark theme CSS into the trame app
                    trame_html.Style('''
                        .v-application, .v-app-bar, body, html {
                            background-color: #000000 !important;
                        }
                        .v-main, .v-container {
                            background-color: transparent !important;
                        }
                    ''')
                    return result
                PyVistaViewer.ui = _dark_ui
                PyVistaViewer._dark_ui_patched = True
        except ImportError:
            pass
