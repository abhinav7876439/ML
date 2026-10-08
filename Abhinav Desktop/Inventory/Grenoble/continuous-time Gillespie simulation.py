import numpy as np
import pandas as pd

def gillespie_inventory(mu=1.0, lam=0.5, theta=0.1, 
                        x0=0, a0=0, xmax=10, 
                        T_max=50, policy_threshold=3, seed=42):
    """
    Continuous-time Gillespie simulation for perishable inventory model.
    """

    np.random.seed(seed)
    t, x, a = 0.0, x0, a0  # time, inventory, action
    records = []

    while t < T_max:
        # Compute event rates
        rate_prod = mu if (a == 1 and x < xmax) else 0.0
        rate_demand = lam if x > 0 else 0.0
        rate_perish = theta * x if x > 0 else 0.0
        
        rates = [rate_prod, rate_demand, rate_perish]
        rate_total = sum(rates)

        if rate_total == 0:
            break  # no events possible

        # Sample next event time
        dt = np.random.exponential(1 / rate_total)
        t += dt

        # Choose event
        r = np.random.rand() * rate_total
        if r < rate_prod:
            event = "production"
            x += 1
        elif r < rate_prod + rate_demand:
            event = "demand"
            x -= 1
        else:
            event = "perish"
            x -= 1

        # Policy: switch if below threshold
        if x <= policy_threshold and a == 0:
            a_new = 1
        elif x > policy_threshold and a == 1:
            a_new = 0
        else:
            a_new = a

        records.append({
            "time": t,
            "inventory": x,
            "prev_action": a,
            "action": a_new,
            "event": event,
            "rate_total": rate_total
        })

        a = a_new

    return pd.DataFrame(records)


# Example run
df = gillespie_inventory(mu=1.0, lam=0.8, theta=0.2, xmax=10, policy_threshold=3, T_max=60)
print(df.head(15))
print("\nFinal state:", df.iloc[-1].to_dict())
