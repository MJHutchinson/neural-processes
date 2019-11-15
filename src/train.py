from src.utils import plot_functions


def train(model, hyperparameters, datagen, optimizer):
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

    EPOCHS = hyperparameters['EPOCHS']
    PLOT_AFTER = hyperparameters['PLOT_AFTER']

    for epoch in range(EPOCHS):
        # Train dataset
        data_train = datagen.generate_curves()
        x_context = data_train.query[0][0].contiguous()
        y_context = data_train.query[0][1].contiguous()
        x_target = data_train.query[1].contiguous()
        y_target = data_train.target_y.contiguous()
        
        optimizer.zero_grad()
        
        y_target_mu, y_target_sigma, log_pred, kl_target_context, loss = model.forward(x_context, y_context, x_target, y_target)
        loss.backward()
        optimizer.step()
        
        print('ITERATION ', epoch, "LOSS ", loss)

        if epoch % PLOT_AFTER == 0:
            plot_functions(x_target, y_target, x_context, y_context, y_target_mu, y_target_sigma)

    return y_target_mu, y_target_sigma, log_pred, kl_target_context, loss
