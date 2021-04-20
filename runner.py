import torch as tc

## need to adapt this to the lm use-case still.
class Runner:
    def __init__(self, max_epochs, verbose=True):
        self.max_epochs = max_epochs
        self.verbose = verbose

    # this needs to be fixed up further to deal with variable length of the sequences
    # and the fact that the padding should not be counted in the loss.
    def train_epoch(self, model, train_dataloader, optimizer, device, criterion):
        for batch_idx, (X, Y) in enumerate(train_dataloader, 1):
            X, Y = X.to(device), Y.to(device)

            # Forward
            logits = model(X)
            loss = criterion(logits.view(-1, model.vocab_size), Y.view(-1, model.vocab_size))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and batch_idx % 100 == 0:
                loss = loss.item()
                print("batch: {}... loss: {}".format(batch_idx, loss))

        return

    # this needs to be fixed up further to deal with variable length of the sequences
    # and the fact that the padding should not be counted in the loss/other metrics.
    def evaluate_epoch(self, model, dataloader, device, criterion):
        num_test_tokens = 0
        test_loss, correct = 0, 0
        with tc.no_grad():
            for X, Y in dataloader:
                X, Y = X.to(device), Y.to(device)

                logits = model(X)
                loss = criterion(logits.view(-1, model.vocab_size), Y.view(-1, model.vocab_size))

                test_loss += X.shape[0] * X.shape[1] * loss.item()
                correct += (logits.argmax(1) == Y).type(tc.float).sum().item()
                num_test_tokens += X.shape[0] * X.shape[1]

        test_loss /= num_test_tokens
        correct /= num_test_tokens
        return {
            "accuracy": correct,
            "loss": test_loss
        }

    def run(self, model, train_dataloader, test_dataloader, device, criterion, optimizer):
        print('running!!')

        for epoch in range(1, self.max_epochs+1):
            if self.verbose:
                print(f"Epoch {epoch}\n-------------------------------")

            model.train() # turn batchnorm, dropout, etc. to train mode.
            self.train_epoch(model, train_dataloader, optimizer, device, criterion)

            model.eval()  # turn batchnorm, dropout, etc. to eval mode.
            test_eval_dict = self.evaluate_epoch(model, test_dataloader, device, criterion)
            test_accuracy = test_eval_dict['accuracy'] * 100
            test_loss = test_eval_dict['loss']
            if self.verbose:
                print(f"Test Error: \n Accuracy: {test_accuracy:>0.1f}%, Avg loss: {test_loss:>8f}\n")

            if epoch % 10 == 0:
                tc.save(model.state_dict(), "model.pth")
                tc.save(optimizer.state_dict(), "optimizer.pth")
