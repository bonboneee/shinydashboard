#from shiny.express import input, render, ui
#
#ui.page_opts(title="Page title")
#
#with ui.sidebar():
#    ui.input_selectize(
#        "var", "변수를 선택해주세요!",
#        choices=["bill_length_mm", "body_mass_g", "bill_depth_mm"]
#    )
#
#    @render.plot # 데코레이터
#    def hist():
#        from matplotlib import pyplot as plt
#        from palmerpenguins import load_penguins
#
#        df = load_penguins()
#        df[input.var()].hist(grid=False) # 인풋에서 선택한 var
#        plt.xlabel(input.var())
#        plt.ylabel("count")
#


