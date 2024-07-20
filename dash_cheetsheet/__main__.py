# application
from dash import Dash

# components
from dash import html, dcc
import dash_bootstrap_components as dbc

# callback
from collections import defaultdict
from dash import ctx, Output, Input, State, ALL, MATCH, no_update
from dash.exceptions import PreventUpdate

# graph
import plotly.express as px
import plotly.graph_objects as go

# others
from typing import Callable
from enum import Enum, auto
from time import sleep


class Application:
    def __init__(self):
        self.app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )

    def setup_components(self):

        class ButtonR1C1:

            class State(Enum):
                ready = 'ready'
                ongoing = ' calculating...'  # first ' ' is space from spinner

            def __init__(self, app, corresponding_command: tuple[Callable, tuple, dict]):
                # register arguments
                self.app = app
                self.corresponding_command = corresponding_command
                # create components
                self.spinner = dbc.Spinner(spinner_style={'display': 'none'}, size='sm')
                self.text = html.Span('ready')
                self.button = dbc.Button([self.spinner, self.text], className="m-2")  # NOTE: Set Margin of all side to 2
                # setup callback
                self.setup_callback()
                # add them to object. use this when add to layout.
                self.obj = html.Div([self.button], className="d-grid")  # NOTE: Use flex grid system in Container.

            def setup_callback(self):
                app = self.app

                # disable when clicked
                @app.callback(
                    output=dict(
                        style=Output(self.spinner, 'spinner_style'),
                        disable_button=Output(self.button, 'disabled'),
                        text=Output(self.text, 'children'),
                    ),
                    inputs=dict(
                        _=Input(self.button, 'n_clicks'),
                    ),
                    state=dict(
                        style=State(self.spinner, 'spinner_style'),
                    ),
                    prevent_initial_call=True,
                )
                def turn_disable(style, **_):
                    output = defaultdict(lambda: no_update)

                    output.update(dict(disable_button=True))
                    output.update(dict(text=self.State.ongoing.value))

                    if style is None:
                        pass
                    else:
                        if 'display' in style.keys():
                            style.pop('display')
                    output.update(dict(style=style))

                    return output

                # process something and re-enable
                @app.callback(
                    Output(self.spinner, 'spinner_style', allow_duplicate=True),
                    Output(self.button, 'disabled', allow_duplicate=True),
                    Output(self.text, 'children', allow_duplicate=True),
                    Input(self.button, 'n_clicks'),
                    State(self.spinner, 'spinner_style'),
                    prevent_initial_call=True,
                )
                def process_something(_, style):
                    try:
                        # do processing
                        f, args, kwargs = self.corresponding_command
                        f(*args, **kwargs)
                    except Exception as e:
                        print(e)
                    if style is None:
                        style = {'display': 'none'}
                    else:
                        style.update({'display': 'none'})
                    return style, False, self.State.ready

        def some_function():
            sleep(3)

        self.button_r1c1: ButtonR1C1 = ButtonR1C1(
            self.app,
            corresponding_command=(some_function, (), {})
        )

        self.button_r1c2: ButtonR1C1 = ButtonR1C1(
            self.app,
            corresponding_command=(some_function, (), {})
        )

        self.button_r2c1: ButtonR1C1 = ButtonR1C1(
            self.app,
            corresponding_command=(some_function, (), {})
        )

        self.button_r2c2: ButtonR1C1 = ButtonR1C1(
            self.app,
            corresponding_command=(some_function, (), {})
        )

    def setup_layout(self):
        container = dbc.Container([
            dbc.Row([
                dbc.Col(self.button_r1c1.obj, width=2),
                dbc.Col(self.button_r1c2.obj, width=2),
            ]),
            dbc.Row([
                dbc.Col(self.button_r2c1.obj, width=2),
                dbc.Col(self.button_r2c2.obj, width=2),
            ]),
        ])
        self.app.layout = container

    def setup_callback(self):
        app = self.app


if __name__ == '__main__':
    g_application = Application()
    g_application.setup_components()
    g_application.setup_layout()
    g_application.setup_callback()
    g_application.app.run(debug=True)
