import numpy as np, pandas as pd

rng = np.random.default_rng(7)
cities = ["Bengaluru","Mumbai","Delhi","Chennai","Kolkata","Hyderabad",
          "Pune","Ahmedabad","Jaipur","Lucknow","Kochi","Mangaluru"]
base = {"Dengue":6,"Malaria":5,"Typhoid":4,"Jaundice":3,
        "Pneumonia":4,"Common Cold":9,"Chicken pox":2,"Tuberculosis":3}
spikes = {("Chennai","Dengue"):4, ("Mumbai","Typhoid"):3, ("Kolkata","Jaundice"):3.5}

end = pd.Timestamp.today().normalize()
days = pd.date_range(end - pd.Timedelta(days=41), end)
rows = []
for c in cities:
    for d, lam in base.items():
        for i, day in enumerate(days):
            mult = spikes.get((c, d), 1) if i >= len(days) - 7 else 1
            rows.append((day, c, d, rng.poisson(lam * rng.uniform(0.6, 1.4) * mult)))

pd.DataFrame(rows, columns=["date","city","disease","cases"]).to_csv(
    "data/regional_daily.csv", index=False)
print("Done:", len(rows), "rows")