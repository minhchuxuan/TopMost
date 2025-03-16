import torch
from torch import nn
import torch.nn.functional as F

class ECR(nn.Module):
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=3000, stopThr=1e-2):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):
        """Compute optimal transport plan using Sinkhorn algorithm
        
        Parameters:
        -----------
        M : torch.Tensor
            Cost matrix, shape [K, V] (topics x vocabulary)
            
        Returns:
        --------
        torch.Tensor
            ECR loss value
        """
        device = M.device
        
        # Initialize marginal distributions
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)  # Topic distribution
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)  # Word distribution
        
        # Initialize scaling vector
        u = (torch.ones_like(a) / a.size()[0]).to(device)
        
        # Compute kernel
        K = torch.exp(-M * self.sinkhorn_alpha)
        
        # Sinkhorn iterations
        err = 1.0
        cpt = 0
        
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))
        
        # Compute transport plan
        transp = u * (K * v.T)
        
        # Compute ECR loss
        loss_ECR = torch.sum(transp * M)
        loss_ECR *= self.weight_loss_ECR
        
        return loss_ECR
class HeadDropout(nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x
        
        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()
        
        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DETM(nn.Module):
    """
    Dynamic Embedded Topic Model with Decomposed Alpha Trajectories (Trend + Seasonal) and MoLE for Seasonal Component,
    using a SINGLE AR(1) PRIOR for the COMBINED Alpha.
    """
    def __init__(self, vocab_size, num_times, train_size, train_time_wordfreq, num_topics=50, train_WE=True, pretrained_WE=None, en_units=800, eta_hidden_size=200, rho_size=300, enc_drop=0.0, eta_nlayers=3, eta_dropout=0.0, delta=0.005, theta_act='relu', device='cpu',
                 num_seasonal_experts=3, alpha_mixing_units=100, decomp_kernel_size=25, head_dropout=0.0, weight_loss_ECR=10.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=800):
        super().__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.num_times = num_times
        self.vocab_size = vocab_size
        self.eta_hidden_size = eta_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = nn.Dropout(enc_drop)
        self.eta_dropout = eta_dropout
        self.delta = delta
        self.train_WE = train_WE
        self.train_size = train_size
        self.rnn_inp = train_time_wordfreq
        self.device = device
        self.theta_act = self.get_activation(theta_act)

        ## Decomposition related
        self.decomp_kernel_size = decomp_kernel_size
        self.decomposition = series_decomp(decomp_kernel_size) # Decomposition block

        ## MoLE related parameters for SEASONAL Alpha Component
        self.num_seasonal_experts = num_seasonal_experts
        self.alpha_mixing_units = alpha_mixing_units
        self.head_dropout = HeadDropout(head_dropout)

        ## define the word embedding matrix \rho
        if self.train_WE:
            self.rho = nn.Linear(self.rho_size, self.vocab_size, bias=False)
        else:
            rho = nn.Embedding(pretrained_WE.size())
            rho.weight.data = torch.from_numpy(pretrained_WE)
            self.rho = rho.weight.data.clone().float().to(self.device)

        ## DLinear for trend component
        self.Linear_Trend = nn.Linear(self.num_times, self.num_times)
        
        ## DLinear for seasonal component
        self.Linear_Seasonal = nn.Linear(self.num_times, self.num_times)
        
        ## Variational parameters for alpha (parameterized via decomposition)
        self.mu_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))

        ## Mixing network for seasonal component
        self.alpha_mixing_net_seasonal = nn.Sequential(
            nn.Linear(1, self.alpha_mixing_units),  # Input is time step index
            nn.ReLU(),
            nn.Linear(self.alpha_mixing_units, self.num_seasonal_experts),
            self.head_dropout,
            nn.Softmax(dim=-1)  # Output weights for experts
        )
        self.weight_loss_ECR = weight_loss_ECR
        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)
        self.beta_temp = 0.2  # Temperature for softmax in get_beta
        ## Expert parameters for seasonal component
        self.mu_q_alpha_seasonal_experts = nn.Parameter(torch.randn(self.num_seasonal_experts, self.num_topics, self.num_times, self.rho_size))
        self.logsigma_q_alpha_seasonal_experts = nn.Parameter(torch.randn(self.num_seasonal_experts, self.num_topics, self.num_times, self.rho_size))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(self.vocab_size + self.num_topics, en_units),
            self.theta_act,
            nn.Linear(en_units, en_units),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(en_units, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(en_units, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)

        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.alpha_fc_mean = nn.Linear(self.rho_size, self.rho_size)
        self.alpha_fc_logvar = nn.Linear(self.rho_size, self.rho_size)
    def pairwise_euclidean_distance(self, x, y):
        """Compute pairwise Euclidean distances between two sets of embeddings
        
        Parameters:
        -----------
        x : torch.Tensor
            First set of embeddings, shape [n, d]
        y : torch.Tensor
            Second set of embeddings, shape [m, d]
            
        Returns:
        --------
        torch.Tensor
            Pairwise squared Euclidean distances, shape [n, m]
        """
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1)
        dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())
        # Ensure no negative distances due to numerical issues
        dist = torch.clamp(dist, min=0.0)
        return dist
    def get_loss_ECR(self, alpha=None):
        """Calculate ECR loss for current time step topic embeddings
        """
        if alpha is None:
            alpha, _ = self.get_alpha()
        
        # Get word embeddings
        if self.train_WE:
            word_embeddings = self.rho.weight.data.to(self.device)
        else:
            word_embeddings = self.rho
        
        # Initialize ECR loss
        total_ecr_loss = 0.0
        
        # For each time step, compute ECR loss
        for t in range(alpha.size(0)):
            topic_embeddings_t = alpha[t]  # [K, rho_size]
            
            # Compute cost matrix: [K, V]
            cost = self.pairwise_euclidean_distance(topic_embeddings_t, word_embeddings)
            
            # Apply Sinkhorn algorithm to compute ECR loss
            ecr_loss = self.ECR(cost)
            total_ecr_loss += ecr_loss
        
        # Average over time steps
        total_ecr_loss /= alpha.size(0)
        
        return total_ecr_loss
    def get_activation(self, act):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'softplus': nn.Softplus(),
            'rrelu': nn.RReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'glu': nn.GLU(),
        }

        if act in activations:
            act = activations[act]
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl
    
    def get_alpha(self):
        """
        Get alpha using efficient vectorized operations with decomposition into trend and seasonal components
        with DLinear processing and MoLE for seasonal components
        """
        device = self.device
        kl_alpha = []
        
        # Initialize tensor to store all alphas
        alphas = torch.zeros((self.num_times, self.num_topics, self.rho_size)).to(device)
        
        # Add FC layers to compute mean and variance for KL
        self.alpha_fc_mean = nn.Linear(self.rho_size, self.rho_size).to(device)
        self.alpha_fc_logvar = nn.Linear(self.rho_size, self.rho_size).to(device)
        
        # Handle first timestep
        alpha_0 = self.mu_q_alpha[:, 0, :]  # Shape: [num_topics, rho_size]
        alphas[0] = alpha_0
        
        # Calculate KL for first timestep
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        p_logsigma_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        
        # Get mean and logvar from the first alpha through FC layers
        q_mu_0 = self.alpha_fc_mean(alpha_0)
        q_logsigma_0 = self.alpha_fc_logvar(alpha_0)
        
        kl_0 = self.get_kl(q_mu_0, q_logsigma_0, p_mu_0, p_logsigma_0)
        kl_alpha.append(kl_0)
        
        # Process remaining timesteps in a single loop, vectorizing operations per timestep
        for t in range(1, self.num_times):
            # Get complete alpha series up to this timestep
            alpha_series = alphas[:t].permute(1, 0, 2)  # [num_topics, t, rho_size]
            
            # Apply decomposition to all topics at once
            seasonal_comp = []
            trend_comp = []
            
            for k in range(self.num_topics):
                # Process each topic, but all dimensions at once
                topic_series = alpha_series[k].unsqueeze(0)  # [1, t, rho_size]
                seas, trend = self.decomposition(topic_series)
                seasonal_comp.append(seas.squeeze(0))
                trend_comp.append(trend.squeeze(0))
            
            seasonal_comp = torch.stack(seasonal_comp)  # [num_topics, t, rho_size]
            trend_comp = torch.stack(trend_comp)  # [num_topics, t, rho_size]
            
            # Get mixture of experts weights for this timestep
            mixing_weights = self.alpha_mixing_net_seasonal(torch.tensor([[t/self.num_times]]).float().to(device))
            
            # Apply mixture of experts for seasonal component
            seasonal_mixture = torch.zeros(self.num_topics, self.rho_size).to(device)
            kl_seasonal = torch.zeros(self.num_topics).to(device)
            
            for e in range(self.num_seasonal_experts):
                expert_mu = self.mu_q_alpha_seasonal_experts[e, :, t, :]
                expert_logsigma = self.logsigma_q_alpha_seasonal_experts[e, :, t, :]
                expert_sample = self.reparameterize(expert_mu, expert_logsigma)
                
                # Weight contribution from this expert
                seasonal_mixture += mixing_weights[0, e] * expert_sample
                
                # KL for this expert (using FC layers to compute mean and var)
                p_mu_seasonal = torch.zeros_like(expert_mu).to(device)
                p_logsigma_seasonal = torch.zeros_like(expert_logsigma).to(device)
                
                # Process the expert sample through FC layers
                q_mu_seasonal = self.alpha_fc_mean(expert_sample)
                q_logsigma_seasonal = self.alpha_fc_logvar(expert_sample)
                
                kl_e = self.get_kl(q_mu_seasonal, q_logsigma_seasonal, p_mu_seasonal, p_logsigma_seasonal)
                kl_seasonal += mixing_weights[0, e] * kl_e
            
            # Get trend component and add both components
            trend_component = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])
            combined_alpha = trend_component + seasonal_mixture
            alphas[t] = combined_alpha
            
            # Calculate KL divergence for the combined alpha with AR(1) prior
            p_mu_t = alphas[t-1]  # Previous alpha serves as prior mean
            p_logsigma_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            
            # Process the combined alpha through FC layers to get mean and var for KL
            q_mu_t = self.alpha_fc_mean(combined_alpha)
            q_logsigma_t = self.alpha_fc_logvar(combined_alpha)
            
            kl_t = self.get_kl(q_mu_t, q_logsigma_t, p_mu_t, p_logsigma_t)
            
            # Total KL for this timestep
            kl_alpha.append(kl_t + kl_seasonal)
        
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)

        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()

        return etas, kl_eta

    def get_theta(self, bows, times, eta=None): ## amortized inference
        """Returns the topic proportions.
        """
        normalized_bows = bows / bows.sum(1, keepdims=True)

        if eta is None and self.training is False:
            eta, kl_eta = self.get_eta(self.rnn_inp)

        eta_td = eta[times]
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))

        if self.training:
            return theta, kl_theta
        else:
            return theta

    @property
    def word_embeddings(self):
        return self.rho.weight

    @property
    def topic_embeddings(self):
        alpha, _ = self.get_alpha()
        return alpha

    def get_beta(self, alpha=None):
        """Returns the topic matrix \beta of shape T x K x V using distance-based approach from ECRTM
        """
        if alpha is None and self.training is False:
            alpha, kl_alpha = self.get_alpha()
        
        # Initialize output tensor for beta: time x topics x vocabulary
        beta = torch.zeros(alpha.size(0), alpha.size(1), self.vocab_size).to(self.device)
        
        # For each time step, compute beta using distance-based approach
        for t in range(alpha.size(0)):
            # Extract topic embeddings for this time step: [K, rho_size]
            topic_embeddings_t = alpha[t]
            
            # Compute pairwise distances between topic embeddings and word embeddings
            # For trainable embeddings, self.rho has weights we can access
            if self.train_WE:
                word_embeddings = self.rho.weight.data.to(self.device)
            else:
                word_embeddings = self.rho
            
            # Calculate Euclidean distances: [K, V]
            dist = self.pairwise_euclidean_distance(topic_embeddings_t, word_embeddings)
            
            # Convert distances to probabilities with softmax and temperature
            beta_t = F.softmax(-dist / 0.2, dim=0)  # Using 0.2 as beta_temp
            beta[t] = beta_t
            
        return beta
    def get_NLL(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = torch.log(loglik + 1e-12)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def forward(self, bows, times):
        bsz = bows.size(0)
        coeff = self.train_size / bsz
        
        # Get eta and theta (keep existing code)
        eta, kl_eta = self.get_eta(self.rnn_inp)
        theta, kl_theta = self.get_theta(bows, times, eta)
        kl_theta = kl_theta.sum() * coeff
        
        # Get alpha and beta
        alpha, kl_alpha = self.get_alpha()
        beta = self.get_beta(alpha)
        
        # Compute NLL (reconstruction loss)
        beta_t = beta[times]
        nll = self.get_NLL(theta, beta_t, bows)
        nll = nll.sum() * coeff
        

        loss_ECR = self.get_loss_ECR(alpha)

        
        # Combine losses
        loss = nll + kl_eta + kl_theta + kl_alpha + loss_ECR
        
        # Return results dictionary with all loss components
        rst_dict = {
            'loss': loss,
            'nll': nll,
            'kl_eta': kl_eta,
            'kl_theta': kl_theta,
            'kl_alpha': kl_alpha,
            'loss_ECR': loss_ECR
        }
        
        return rst_dict

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \\eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
