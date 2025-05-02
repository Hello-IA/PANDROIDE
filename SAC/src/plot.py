


def plot_learning_curve(logger_critic_loss_1, logger_actor_loss, logger_reward, logger_nb_steps, save_path=None):
    """
    Plot learning curves from the logger data
    
    Args:
        logger: The logger object from the algorithm
        save_path: Optional path to save the figure
    """

    # Create a figure with multiple subplots
    

    plt.plot(logger_nb_steps, logger_actor_loss)
    plt.title('Actor Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()

    
    plt.plot(logger_nb_steps, logger_critic_loss_1, label='Critic 1')
    plt.title('Critic Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

    plt.plot(range(len(logger_reward)), logger_reward, label='Reward')
    plt.title('Reward')
    plt.xlabel('Steps')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.show()