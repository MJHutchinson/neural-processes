from src.utils import plot_functions
from torch import gather


def train(
    model,
    hyperparameters,
    datagen,
    datagen_test,
    optimizer,
    save=False,
    experiment_name=None,
):
    """
    Trains the NP, given hyperparameters and the anp

    Parameters
    ----------
    hyperparameters : dictionary TODO: We can add in more later
        Keys: EPOCHS, PLOT_AFTER
        Values: int, int,

    datagen : object
        object that generates data in the form specified as 
        ((x_context, y_context), x_target)
        y_target
        num_context
        total_num
    
    optimizer: torch.nn.Module
        
    return : 
    
    y_target_mu: torch.Tensor, final
        Shape (batch_size, num_target, y_dim)
    
    y_target_sigma: torch.Tensor, final
        Shape (batch_size, num_target, y_dim)
    
    log_pred: torch.Tensor, final
        Shape (batch_size, num_target)

    kl_target_context: torch.Tensor, final
        Shape (batch_size, num_target)

    loss: torch.Tensor, final
        Shape (1,)
    """

    EPOCHS = hyperparameters["EPOCHS"]
    PLOT_AFTER = hyperparameters["PLOT_AFTER"]

    for epoch in range(EPOCHS):
        # Train dataset
        data_train = datagen.generate_curves()
        x_context = data_train.query[0][0].contiguous()
        x_context, x_context_sorted_indices = x_context.sort(1)
        y_context = data_train.query[0][1].contiguous()
        y_context = gather(y_context, 1, x_context_sorted_indices)
        x_target = data_train.query[1].contiguous()
        x_target, x_target_sorted_indices = x_target.sort(1)
        y_target = data_train.target_y.contiguous()
        y_target = gather(y_target, 1, x_target_sorted_indices)

        optimizer.zero_grad()

        y_target_mu, y_target_sigma, log_pred, kl_target_context, loss = model.forward(
            x_context, y_context, x_target, y_target
        )
        loss.backward()
        optimizer.step()


        if epoch % PLOT_AFTER == 0:
            plot_functions(
                x_target,
                y_target,
                x_context,
                y_context,
                y_target_mu,
                y_target_sigma,
                save=save,
                experiment_name=experiment_name + "_train",
                iter=epoch,
            )
            data_test = datagen_test.generate_curves()
            x_context = data_test.query[0][0].contiguous()
            y_context = data_test.query[0][1].contiguous()
            x_target = data_test.query[1].contiguous()
            y_target = data_test.target_y.contiguous()
            y_target_mu, y_target_sigma, _, _, _ = model.forward(
                x_context, y_context, x_target, y_target
            )
            plot_functions(
                x_target,
                y_target,
                x_context,
                y_context,
                y_target_mu,
                y_target_sigma,
                save=save,
                experiment_name=experiment_name,
                iter=epoch,
            )
            print(
                f"Iter: {epoch}, log_pred: {log_pred.sum()}, kl_target_context: {kl_target_context.sum()}, loss: {loss.sum()}"
            )

    return y_target_mu, y_target_sigma, log_pred, kl_target_context, loss

