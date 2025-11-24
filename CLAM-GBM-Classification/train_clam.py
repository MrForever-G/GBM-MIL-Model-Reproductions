import torch
from evaluation import eval_model
from utils import get_logger


def train(model, train_loader, test_loader, loss_func, optimizer, scheduler, args):
    device = args.device
    model.to(device)

    logger = args.logger
    writer = args.writer
    num_epochs = args.num_epochs

    step = 0

    for epoch in range(num_epochs):
        model.train()

        loss_sum = 0.0
        n_samples = 0

        # ...
        for data in train_loader:
            # move the whole PyG Data object to CUDA
            data = data.to(device)

            # labels
            y = data.y

            # forward
            logits = model(data)

            # CE expects shape [B,C], but our bag is batch=1 → expand dimension
            loss = loss_func(logits.unsqueeze(0), y)

            # backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # scheduler
            if scheduler:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("LR", lr, step)
                step += 1

            loss_sum += loss.item()
            n_samples += 1

        avg_loss = loss_sum / max(1, n_samples)

        # evaluation (patient-level AUC)
        eval_train = eval_model(model, train_loader, device)
        eval_test = eval_model(model, test_loader, device)

        # log
        logger.info(
            "Epoch[%d/%d], loss: %.4f",
            epoch + 1, num_epochs, avg_loss
        )
        logger.info(
            "Epoch[%d/%d], train AUC: %.4f(%.4f-%.4f), test AUC: %.4f(%.4f-%.4f)",
            epoch + 1, num_epochs,
            eval_train[0], eval_train[1], eval_train[2],
            eval_test[0], eval_test[1], eval_test[2]
        )

        # tensorboard
        writer.add_scalar("LOSS/train", avg_loss, epoch + 1)
        writer.add_scalars("AUC", {
            "train": eval_train[0],
            "test": eval_test[0]
        }, epoch + 1)

    # save model
    if args.savedisk:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": num_epochs,
        }
        torch.save(state, f"{args.model_path}/model.pt")
