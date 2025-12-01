import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    return pd, plt


@app.cell
def _(pd):
    dat = pd.read_csv("data/raw/final_leaderboard.csv")

    myscore = dat.loc[dat["TeamName"] == "Bla Blub", "Score"].iloc[0]

    dat.head()
    return dat, myscore


@app.cell
def _(dat, plt):
    plt.hist(dat["Score"], bins=100, density=True, edgecolor="black")
    plt.title("Distribution Leaderboard Scores")
    plt.xlabel("Scores")
    plt.ylabel("Density")
    plt.savefig("data/plots/dis_lead_all.png")
    plt.show()
    return


@app.cell
def _(dat, myscore, plt):
    subset = dat.loc[dat["Score"] > 0.9, "Score"]

    plt.hist(subset, bins=200, density=True, edgecolor="black")
    plt.axvline(myscore, linestyle="--", linewidth=2, color="red")
    plt.text(myscore - 0.003, plt.ylim()[1] * 0.9, "My Score", color="red", ha="center", va="bottom")
    plt.title("Distribution Leaderboard Scores above 0.9")
    plt.xlabel("Scores")
    plt.ylabel("Density")
    plt.savefig("data/plots/dis_lead_above.png")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
