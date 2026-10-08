"""Reproducible GA-optimized first-order Takagi–Sugeno classifier."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ['max_temp_c', 'min_temp_c', 'relative_humidity_pct', 'duration_days', 'temp_anomaly_c']
CLASSES = ['Low', 'Moderate', 'High', 'Extreme']

def validate(frame, target=False):
    missing = set(FEATURES + (['risk'] if target else [])) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    x = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Features must be finite numbers; impute missing data before training.')
    if ((x[:, 2] < 0) | (x[:, 2] > 100)).any() or (x[:, 3] < 1).any():
        raise ValueError('Humidity must be 0–100 and duration at least 1 day.')
    if (x[:, 1] > x[:, 0]).any():
        raise ValueError('Minimum temperature cannot exceed maximum temperature.')
    if target and not frame.risk.isin(CLASSES).all():
        raise ValueError(f'Risk labels must be one of {CLASSES}')
    return x

def synthetic(n=1200, seed=42):
    """Invented label mechanism for software demonstration, not clinical thresholds."""
    rng = np.random.default_rng(seed)
    maximum = rng.uniform(28, 49, n)
    minimum = maximum - rng.uniform(5, 16, n)
    humidity = rng.uniform(15, 95, n)
    duration = rng.integers(1, 11, n)
    anomaly = rng.uniform(-1, 10, n)
    score = (maximum - 32) * .65 + (minimum - 23) * .35 + (humidity - 45) * .045 + duration * .42 + anomaly * .4 + rng.normal(0, 1.1, n)
    frame = pd.DataFrame(np.column_stack([maximum, minimum, humidity, duration, anomaly]), columns=FEATURES)
    frame['risk'] = np.array(CLASSES)[np.digitize(score, [4, 9, 14])]
    return frame

class NeuroFuzzy:
    """Gaussian product rules, normalized firing, trainable softmax consequents.

    Each rule has its own center and width per input. First-order consequents
    produce class logits: sum_r w_r(x) * (b_rc + a_rc @ x).
    Logistic regression learns those coefficients using cross-entropy and L2.
    """
    def __init__(self, centers, widths, regularization=1.0):
        self.centers = np.asarray(centers).copy()
        self.widths = np.asarray(widths).copy()
        self.regularization = regularization

    def firing(self, x):
        logits = -.5 * (((x[:, None, :] - self.centers) / self.widths) ** 2).sum(axis=2)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)

    def design(self, x):
        augmented = np.column_stack([np.ones(len(x)), x])
        return (self.firing(x)[:, :, None] * augmented[:, None, :]).reshape(len(x), -1)

    def fit(self, x, y):
        if (self.widths <= 0).any():
            raise ValueError('Gaussian widths must be positive.')
        self.head = LogisticRegression(C=self.regularization, max_iter=700, class_weight='balanced')
        self.head.fit(self.design(x), y)
        return self

    def predict(self, x):
        return self.head.predict(self.design(x))

    def predict_proba(self, x):
        return self.head.predict_proba(self.design(x))

def optimize(x, y, xv, yv, rules=6, population=12, generations=10, seed=42):
    if population < 4 or generations < 1 or rules < 1:
        raise ValueError('Need population >= 4, generations >= 1 and rules >= 1.')
    rng = np.random.default_rng(seed)
    centers = KMeans(n_clusters=rules, n_init=10, random_state=seed).fit(x).cluster_centers_
    d = centers.size
    initial = np.r_[centers.ravel(), np.zeros(d), 0.0]
    lower = np.r_[np.full(d, -4.), np.full(d, -1.8), -2.]
    upper = np.r_[np.full(d, 4.), np.full(d, 1.2), 2.]
    def decode(g):
        return NeuroFuzzy(g[:d].reshape(centers.shape), np.exp(g[d:2*d]).reshape(centers.shape), 10 ** g[-1])
    pool = np.clip(initial + rng.normal(0, .25, (population, len(initial))), lower, upper)
    pool[0] = initial
    history = []
    best, best_score = initial.copy(), -1.
    for generation in range(generations):
        scores = np.array([f1_score(yv, decode(g).fit(x, y).predict(xv), labels=CLASSES, average='macro', zero_division=0) for g in pool])
        winner = int(scores.argmax())
        if scores[winner] > best_score:
            best, best_score = pool[winner].copy(), float(scores[winner])
        history.append({'generation': generation + 1, 'best_validation_macro_f1': best_score, 'population_mean': float(scores.mean())})
        next_pool = [best.copy(), pool[np.argsort(scores)[-2]].copy()]
        def tournament():
            candidates = rng.choice(population, 3, replace=False)
            return pool[candidates[np.argmax(scores[candidates])]]
        while len(next_pool) < population:
            a, b = tournament(), tournament()
            mixing = rng.random(len(a))
            child = mixing * a + (1 - mixing) * b
            child += (rng.random(len(child)) < .15) * rng.normal(0, .2, len(child))
            next_pool.append(np.clip(child, lower, upper))
        pool = np.array(next_pool)
        print(f'Generation {generation+1}/{generations}: validation macro-F1 {best_score:.4f}', flush=True)
    return decode(best), NeuroFuzzy(centers, np.ones_like(centers)), history

def train(frame, out, rules=6, population=12, generations=10, seed=42, chronological=False):
    x = validate(frame, target=True)
    y = frame.risk.to_numpy()
    if set(y) != set(CLASSES) or frame.risk.value_counts().min() < 10:
        raise ValueError('Training requires all four classes with at least 10 rows per class.')
    if chronological:
        dates = pd.to_datetime(frame['date'], errors='raise')
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError('Chronological mode requires unique ascending dates.')
        masks = [dates < '2022-01-01', (dates >= '2022-01-01') & (dates < '2024-01-01'), dates >= '2024-01-01']
        if any(not m.any() for m in masks): raise ValueError('Empty chronological partition.')
        xa, xv, xe = [x[m] for m in masks]
        ya, yv, ye = [y[m] for m in masks]
        if any(set(part) != set(CLASSES) for part in [ya, yv, ye]):
            raise ValueError('Each chronological partition must contain all four classes.')
    else:
        xt, xe, yt, ye = train_test_split(x, y, test_size=.2, stratify=y, random_state=seed)
        xa, xv, ya, yv = train_test_split(xt, yt, test_size=.25, stratify=yt, random_state=seed)
    scaler = StandardScaler().fit(xa)
    a, v, e = scaler.transform(xa), scaler.transform(xv), scaler.transform(xe)
    selected, fixed, history = optimize(a, ya, v, yv, rules, population, generations, seed)
    # Preserve the training-only scaler: optimized premises live in these coordinates.
    combined = np.vstack([a, v]); labels = np.r_[ya, yv]
    models = {'GA Neuro-Fuzzy': selected.fit(combined, labels), 'Fixed Neuro-Fuzzy': fixed.fit(combined, labels),
              'Logistic Regression': LogisticRegression(max_iter=700, class_weight='balanced').fit(combined, labels),
              'Random Forest': RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=seed).fit(combined, labels)}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, model in models.items():
        pred = model.predict(e)
        report[name] = {'accuracy': accuracy_score(ye, pred), 'macro_f1': f1_score(ye, pred, labels=CLASSES, average='macro', zero_division=0),
                       'classification_report': classification_report(ye, pred, labels=CLASSES, output_dict=True, zero_division=0),
                       'confusion_matrix': confusion_matrix(ye, pred, labels=CLASSES).tolist()}
    metadata = {'seed': seed, 'rows': len(frame), 'split_sizes': {'train': len(a), 'validation': len(v), 'test': len(e)},
                'features': FEATURES, 'class_order': CLASSES, 'rules': rules, 'population': population, 'generations': generations,
                'split_method': 'chronological: train before 2022, validation 2022–2023, test 2024 onward' if chronological else 'stratified random',
                'data_origin': frame.attrs.get('origin', 'user supplied; provenance must be verified')}
    (out / 'metrics.json').write_text(json.dumps({'metadata': metadata, 'models': report}, indent=2))
    pd.DataFrame(history).to_csv(out / 'ga_history.csv', index=False)
    with (out / 'model.pkl').open('wb') as handle:
        pickle.dump({'scaler': scaler, 'model': selected, 'metadata': metadata}, handle)
    centers_native = scaler.inverse_transform(selected.centers)
    pd.DataFrame(centers_native, columns=FEATURES).to_csv(out / 'rule_centers.csv', index_label='rule')
    pd.DataFrame(selected.widths * scaler.scale_, columns=FEATURES).to_csv(out / 'rule_widths.csv', index_label='rule')
    pred = selected.predict(e)
    pd.DataFrame({'actual': ye, 'predicted': pred}).to_csv(out / 'test_predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].plot([h['generation'] for h in history], [h['best_validation_macro_f1'] for h in history], marker='o')
    axes[0].set(xlabel='Generation', ylabel='Validation macro-F1', title='GA selection (validation only)')
    cm = confusion_matrix(ye, pred, labels=CLASSES)
    axes[1].imshow(cm, cmap='Blues')
    for i in range(4):
        for j in range(4): axes[1].text(j, i, str(cm[i, j]), ha='center', va='center')
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted', ylabel='Actual', title='Untouched test set')
    fig.savefig(out / 'evaluation.png', dpi=170); plt.close(fig)
    return report

def predict_file(model_path, frame):
    # Load only trusted locally produced model files; pickle can execute code.
    with open(model_path, 'rb') as handle: bundle = pickle.load(handle)
    x = bundle['scaler'].transform(validate(frame))
    model = bundle['model']
    result = frame.copy(); result['predicted_risk'] = model.predict(x)
    for label, values in zip(model.head.classes_, model.predict_proba(x).T):
        result[f'probability_{label.lower()}'] = values
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo'); demo.add_argument('--rows', type=int, default=1200)
    training = sub.add_parser('train'); training.add_argument('--csv', required=True); training.add_argument('--chronological', action='store_true')
    for p in [demo, training]:
        p.add_argument('--out', default='results'); p.add_argument('--seed', type=int, default=42)
        p.add_argument('--rules', type=int, default=6); p.add_argument('--population', type=int, default=12); p.add_argument('--generations', type=int, default=10)
    prediction = sub.add_parser('predict'); prediction.add_argument('--model', default='results/model.pkl'); prediction.add_argument('--csv', required=True); prediction.add_argument('--out', default='results/predictions.csv')
    args = parser.parse_args()
    if args.command == 'predict':
        output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
        predict_file(args.model, pd.read_csv(args.csv)).to_csv(output, index=False)
    else:
        frame = synthetic(args.rows, args.seed) if args.command == 'demo' else pd.read_csv(args.csv)
        if args.command == 'demo':
            frame.attrs['origin'] = 'synthetic demonstration with invented labels'
            Path(args.out).mkdir(parents=True, exist_ok=True); frame.to_csv(Path(args.out) / 'synthetic_data.csv', index=False)
        result = train(frame, args.out, args.rules, args.population, args.generations, args.seed, getattr(args, 'chronological', False))
        print(json.dumps({k: {m: v[m] for m in ['accuracy', 'macro_f1']} for k, v in result.items()}, indent=2))

if __name__ == '__main__':
    # Ensure models remain loadable when this file is run directly.
    from heatwave import main as entry
    entry()
