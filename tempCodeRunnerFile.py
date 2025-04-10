def produce_samples(net, time_embed, test_data, num_samples=100, num_timesteps=1000, eps=1e-5):
    # Limit samples to speed up the process
    num_samples = min(num_samples, len(test_data))
    test_data = test_data[:num_samples]
    
    # Get image dimensions
    batch_size, height, width = test_data.shape
    
    # Forward process timesteps (0,2)
    forward_ts = torch.linspace(0., 2., num_timesteps)
    forward_states = []
    
    # Generate forward process states
    # Start with flattened images for processing
    x0 = test_data.reshape(batch_size, -1)
    # Store the original shape for visualization
    forward_states.append(test_data.detach().numpy().copy())
    
    with torch.no_grad():
        for t in forward_ts[1:]:
            # Apply noise
            var = (torch.exp(2*t) - 1)/2
            noise = torch.sqrt(var) * torch.randn_like(x0)
            x_noisy = x0 + noise
            
            # Store (reshape back to image for visualization)
            forward_states.append(x_noisy.reshape(batch_size, height, width).detach().numpy().copy())
    
    # Start reverse process with the noisy samples
    x = torch.tensor(forward_states[-1], dtype=torch.float32).reshape(batch_size, -1)
    
    # Reverse process timesteps (2, 0)
    reverse_ts = torch.linspace(2., eps, num_timesteps)
    dt = reverse_ts[1] - reverse_ts[0]
    
    # Store reverse process states
    reverse_states = [x.reshape(batch_size, height, width).numpy().copy()]
    score_approx = []
    
    # Perform reverse SDE
    with torch.no_grad():
        for i in range(len(reverse_ts)-1):
            t = reverse_ts[i]
            t_batch = torch.full((batch_size, 1), t)
            t_embed = time_embed(t_batch)
            
            # Predict score (with flattened x)
            score = net(torch.cat([x, t_embed], dim=1))
            score_approx.append(score.mean().item())
            
            # Update x
            g_t = diffusion(t)
            drift = -(g_t**2) * score
            noise = g_t * torch.randn_like(x) * torch.sqrt(torch.abs(dt))
            x = x + (drift * dt) + noise
            
            # Store (reshape back to image for visualization)
            reverse_states.append(x.reshape(batch_size, height, width).detach().numpy().copy())
    
    return forward_states, reverse_states, score_approx