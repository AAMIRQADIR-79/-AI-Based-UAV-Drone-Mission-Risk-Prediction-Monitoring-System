import sys
sys.path.insert(0, '.')
from utils.alerts import evaluate_alerts, battery_drain_per_km, compute_readiness

print('Testing alerts...')
alerts = evaluate_alerts(
    battery_pct=18, wind_speed=13, signal_pct=25,
    payload_kg=2, distance_km=90, temperature=30,
    risk_score=72, altitude_m=120, humidity_pct=60
)
print(f'Total alerts: {len(alerts)}')
for a in alerts:
    print(f'  [{a.level.upper()}] {a.message}')

print()
print('Testing battery drain...')
drain = battery_drain_per_km(payload_kg=2, wind_speed=8)
print(f'Drain rate: {drain} % per km')

print()
print('Testing readiness score...')
readiness = compute_readiness(72, alerts, nfz_conflicts=1, weather_risk=30)
print('Readiness:', readiness['readiness_score'], '/100')
print('Verdict:', readiness['verdict'])

print()
print('ALL ALERTS OK')