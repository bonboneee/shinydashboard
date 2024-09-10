from shiny.express import ui

with ui.layout_column_wrap(gap="2rem"):
    ui.input_slider("slider1", "Slider 1", min=0, max=100, value=50)
    ui.input_slider("slider2", "Slider 2", min=0, max=100, value=50)


    from shiny.express import ui

with ui.navset_pill(id="tab"):  
    with ui.nav_panel("A"):
        "Panel A content"

    with ui.nav_panel("B"):
        "Panel B content"

    with ui.nav_panel("C"):
        "Panel C content"

    with ui.nav_menu("Other links"):
        with ui.nav_panel("D"):
            "Page D content"

        "----"
        "Description:"
        with ui.nav_control():
            ui.a("Shiny", href="https://shiny.posit.co", target="_blank")


from shiny.express import input, render, ui

ui.page_opts(fillable=True)

with ui.layout_columns():  
    with ui.card():  
        ui.card_header("Card 1 header")
        ui.p("Card 1 body")
        ui.input_slider("slider", "Slider", 0, 10, 5)

    with ui.card():  
        ui.card_header("Card 2 header")
        ui.p("Card 2 body")
        ui.input_text("text", "Add text", "")
@render.text
def text_out():
    return f"Input text: {input.text()}"




from shiny.express import ui

ui.page_opts(title="Page title")

with ui.sidebar():
    "Sidebar content"

"Main content"

