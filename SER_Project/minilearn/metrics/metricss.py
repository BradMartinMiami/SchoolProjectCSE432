# classification metrics from scratch
# week 7

import numpy as np
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred):
    # how many we got right divided by total
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels):
    # rows = actual class, cols = predicted class
    # cell [i,j] = how many times true was i but model said j
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for i in range(len(y_true)):
        # find which row/col each label is
        true_idx = labels.index(y_true[i])
        pred_idx = labels.index(y_pred[i])
        cm[true_idx, pred_idx] += 1

    return cm


def precision(y_true, y_pred, labels, average='macro'):
    # precision = TP / (TP + FP)
    # of all the times we predicted class X, how often was it actually X?
    cm = confusion_matrix(y_true, y_pred, labels)

    per_class = []
    for i in range(len(labels)):
        TP = cm[i, i]
        # column sum = everything we predicted as class i
        col_sum = cm[:, i].sum()
        if col_sum == 0:
            per_class.append(0.0)
        else:
            per_class.append(TP / col_sum)
    per_class = np.array(per_class)

    if average == 'macro':
        return per_class.mean()
    elif average == 'weighted':
        # weight by how many actual samples each class has
        weights = cm.sum(axis=1)
        return (per_class * weights).sum() / weights.sum()
    else:
        return per_class   # per-class array


def recall(y_true, y_pred, labels, average='macro'):
    # recall = TP / (TP + FN)
    # of all the actual class X samples, how many did we catch?
    cm = confusion_matrix(y_true, y_pred, labels)

    per_class = []
    for i in range(len(labels)):
        TP = cm[i, i]
        # row sum = all actual class i samples
        row_sum = cm[i, :].sum()
        if row_sum == 0:
            per_class.append(0.0)
        else:
            per_class.append(TP / row_sum)
    per_class = np.array(per_class)

    if average == 'macro':
        return per_class.mean()
    elif average == 'weighted':
        weights = cm.sum(axis=1)
        return (per_class * weights).sum() / weights.sum()
    else:
        return per_class


def f1_score(y_true, y_pred, labels, average='macro'):
    # F1 = 2 * (P * R) / (P + R)
    # harmonic mean of precision and recall
    p = precision(y_true, y_pred, labels, average=None)
    r = recall(y_true, y_pred, labels, average=None)

    per_class = []
    for i in range(len(labels)):
        if p[i] + r[i] == 0:
            per_class.append(0.0)
        else:
            f1 = 2 * p[i] * r[i] / (p[i] + r[i])
            per_class.append(f1)
    per_class = np.array(per_class)

    if average == 'macro':
        return per_class.mean()
    elif average == 'weighted':
        cm = confusion_matrix(y_true, y_pred, labels)
        weights = cm.sum(axis=1)
        return (per_class * weights).sum() / weights.sum()
    else:
        return per_class


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    # heatmap with the counts inside each cell
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    # write the numbers in each cell
    # darker cells need white text so you can read them
    threshold = cm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j] > threshold:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def roc_curve(y_true_binary, y_score):
    # ROC for one binary problem
    # returns fpr and tpr arrays you can plot

    y_true_binary = np.array(y_true_binary)
    y_score = np.array(y_score)

    # sort everything by score, highest first
    order = np.argsort(-y_score)
    y_sorted = y_true_binary[order]

    P = (y_true_binary == 1).sum()  # total positives
    N = (y_true_binary == 0).sum()  # total negatives

    # cumulative TPs and FPs as we go down the sorted list
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = tps / P
    fpr = fps / N

    # add a (0,0) point at the start so the curve starts at the origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    return fpr, tpr


def auc(fpr, tpr):
    # area under the ROC curve, using trapezoid rule
    # numpy renamed trapz to trapezoid in 2.0 so check which one exists
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(tpr, fpr)
    else:
        return np.trapz(tpr, fpr)


def roc_auc(y_true, y_proba, labels, average='macro'):
    # multi-class AUC, one-vs-rest
    # for each class, treat it as positive, get its AUC, then average
    y_true = np.array(y_true)

    aucs = []
    for i in range(len(labels)):
        # binary version: 1 if true class is this one, 0 otherwise
        y_bin = (y_true == labels[i]).astype(int)
        # the model's probability for this class
        scores = y_proba[:, i]
        fpr, tpr = roc_curve(y_bin, scores)
        aucs.append(auc(fpr, tpr))
    aucs = np.array(aucs)

    if average == 'macro':
        return aucs.mean()
    else:
        return aucs


def plot_roc_curves(y_true, y_proba, labels, class_names, title='ROC Curves'):
    y_true = np.array(y_true)

    plt.figure(figsize=(8, 6))
    for i in range(len(labels)):
        y_bin = (y_true == labels[i]).astype(int)
        scores = y_proba[:, i]
        fpr, tpr = roc_curve(y_bin, scores)
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={a:.2f})')

    # diagonal line = random guessing
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_report(y_true, y_pred, y_proba, labels, class_names, model_name='Model'):
    # print everything at once so I don't have to call each function
    print(f'=== {model_name} ===')
    print(f'Accuracy:           {accuracy(y_true, y_pred):.4f}')
    print(f'Precision (macro):  {precision(y_true, y_pred, labels):.4f}')
    print(f'Recall (macro):     {recall(y_true, y_pred, labels):.4f}')
    print(f'F1 (macro):         {f1_score(y_true, y_pred, labels):.4f}')
    print(f'F1 (weighted):      {f1_score(y_true, y_pred, labels, average="weighted"):.4f}')
    print(f'AUC (macro):        {roc_auc(y_true, y_proba, labels):.4f}')
    print()

    # per-class breakdown
    p = precision(y_true, y_pred, labels, average=None)
    r = recall(y_true, y_pred, labels, average=None)
    f = f1_score(y_true, y_pred, labels, average=None)
    cm = confusion_matrix(y_true, y_pred, labels)
    support = cm.sum(axis=1)

    print(f'{"class":<12} {"precision":>10} {"recall":>10} {"f1":>10} {"count":>10}')
    for i in range(len(labels)):
        print(f'{class_names[i]:<12} {p[i]:>10.4f} {r[i]:>10.4f} {f[i]:>10.4f} {support[i]:>10d}')