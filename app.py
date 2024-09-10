from shiny.express import input, render, ui
ui.page_opts(title="팔머펭귄 부리 깊이 예측하기!")
with ui.sidebar():
    ui.input_selectize(
        "var", "펭귄 종을 선택해주세요!",
        choices=["Adelie", "Gentoo", "Chinstrap"]
    )
    ui.input_slider("slider1", "부리길이를 입력해주세요!", min=0, max=100, value=50)

    @render.text
    def cal_depth():
        if input.var() == "Adelie":
            y_hat = 0.2 * input.slider1() + 10.56
        elif input.var() == "Chinstrap":
            y_hat = 0.2 * input.slider1() + (10.56 -1.93)
        else:
            y_hat = 0.2 * input.slider1() + (10.56 -5.10)
        return f"부리깊이 예상치: {round(y_hat, 3)}"

@render.plot # 데코레이터
def scatter():
    from matplotlib import pyplot as plt
    import seaborn as sns
    from palmerpenguins import load_penguins
    import pandas as pd

    df = load_penguins()

    # !pip install scikit-learn
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    penguins=df.dropna()

    penguins_dummies = pd.get_dummies(
        penguins, 
        columns=['species'],
        drop_first=True
        )

    # x와 y 설정
    x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
    y = penguins_dummies["bill_depth_mm"]

    # 모델 학습
    model.fit(x, y)

    model.coef_
    model.intercept_

    regline_y=model.predict(x)

    import numpy as np
    index_1=np.where(penguins['species'] == "Adelie")
    index_2=np.where(penguins['species'] == "Gentoo")
    index_3=np.where(penguins['species'] == "Chinstrap")

    plt.rcParams.update({'font.family': 'Malgun Gothic'})
    sns.scatterplot(data=df, 
                    x="bill_length_mm", 
                    y="bill_depth_mm",
                    hue="species")
    plt.plot(penguins["bill_length_mm"].iloc[index_1], regline_y[index_1], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_2], regline_y[index_2], color="black")
    plt.plot(penguins["bill_length_mm"].iloc[index_3], regline_y[index_3], color="black")
    plt.xlabel("부리길이")
    plt.ylabel("부리깊이")