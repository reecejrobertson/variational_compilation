import torch
import tqdm


def training_loop(optimizer, model, scheduler, num_steps, show_progress=True, early_stopping_threshold=1e-10):
    pbar = tqdm.tqdm(range(num_steps), disable=not show_progress)
    previous_loss = torch.inf
    for step in pbar:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"Loss={loss} - LR={scheduler.get_last_lr()[0]}")
        if step > 100 and torch.abs(previous_loss - loss) < early_stopping_threshold:
            print("Early stopping loss difference is smaller than 1e-10")
            break
        previous_loss = loss.clone()
    return model

