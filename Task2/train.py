from nn import MLP
model = MLP(3, [4, 4, 1])
print(model)
print("number of parameters", len(model.parameters()))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

for k in range(200):
    # forward
    y_pred = [model(xi) for xi in xs]
    # calculate loss (mean square error)
    loss = [(yi_pred - yi) ** 2 for yi_pred, yi in zip(y_pred, ys)]
    loss = sum(loss) / len(loss)
    acc = [(y_pred.data > 0) == (yi > 0) for y_pred, yi in zip(y_pred, ys)]
    acc = sum(acc) / len(acc)
    # backward (zero_grad + backward)
    model.zero_grad()
    loss.backward()

    # update
    learning_rate = 0.001
    for p in model.parameters():
        p.data = p.data - learning_rate * p.grad

    if k % 10 == 0:
        print(f"step {k} loss {loss.data}, accuracy {acc*100}%")
