import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_sure(scorenet, samples, sigmas, sigmas_n, lamda, labels=None, anneal_power=2., hook=None):
    # if labels is None:
    #     labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    # used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    # noise = torch.randn_like(samples) * used_sigmas
    # perturbed_samples = samples + noise
    # target = - 1 / (used_sigmas ** 2) * noise
    # scores = scorenet(perturbed_samples, labels)
    # target = target.view(target.shape[0], -1)
    # scores = scores.view(scores.shape[0], -1)
    # loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # DSM_loss =loss.mean(dim=0)

    # eps=0.001
    # b_prime = torch.randn_like(samples)
    # Yptb = samples + b_prime * eps
    # Y = samples
    # var_Y = sigmas_n
    # Yptb_ = scorenet(Yptb, labels)*used_sigmas*(var_Y+eps)
    # Y_= scorenet(Y, labels)*used_sigmas*var_Y
    # Ht = samples.shape[1]  # height of the image
    # Wt = samples.shape[2]  # width  of the image
    # batch = samples.shape[0]
    # divergence = (1.0 / eps)*(b_prime*(Yptb_ - Y_))
    # divergence_sum_Y = (var_Y * divergence).sum()
    # # SURE
    # var_sum_Y = Ht * Wt * 1 * var_Y.sum() / 2.0
    # sure_loss = (1.0 / samples.shape[0]) * (torch.nn.functional.mse_loss(Y, Y_) - var_sum_Y + divergence_sum_Y)

    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    # perturbed_samples = samples+(sigmas_n**2)*(scorenet(samples, labels)*used_sigmas)/sigmas_n + noise

    grad=scorenet(samples, labels) * used_sigmas
    perturbed_samples = samples + sigmas_n * grad + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    DSM_loss =loss.mean(dim=0)#10026

    # sure_loss_1= (1.0 / samples.shape[0]) *(sigmas_n**2)*(scorenet(samples, labels)*used_sigmas)/sigmas_n
    # sure_loss_1 = (1.0 / samples.shape[0]) * sigmas_n * grad
    sure_loss_1 = sigmas_n * grad
    eps = 0.001
    b_prime = torch.randn_like(samples)
    temp_1= samples+sigmas_n*grad
    temp_2= samples+eps*b_prime + sigmas_n*grad
    div=2*(sigmas_n**2)*(1.0 / eps)*b_prime*(temp_1-temp_2)
    sure_loss_1 = sure_loss_1.view(sure_loss_1.shape[0], -1)
    div = div.view(div.shape[0], -1)
    sure_loss = (sure_loss_1 + div).sum(dim=-1)
    sure_loss=sure_loss.mean(dim=0) #342
    # eps=0.001
    # b_prime = torch.randn_like(samples)
    # Yptb = samples + b_prime * eps
    # Y = samples
    # var_Y = sigmas_n
    # Yptb_ = scorenet(Yptb, labels)*used_sigmas*(var_Y+eps)
    # Y_= scorenet(Y, labels)*used_sigmas*var_Y
    # Ht = samples.shape[1]  # height of the image
    # Wt = samples.shape[2]  # width  of the image
    # batch = samples.shape[0]
    # divergence = (1.0 / eps)*(b_prime*(Yptb_ - Y_))
    # divergence_sum_Y = (var_Y * divergence).sum()
    # # SURE
    # var_sum_Y = Ht * Wt * 1 * var_Y.sum() / 2.0
    # sure_loss = (1.0 / samples.shape[0]) * (torch.nn.functional.mse_loss(Y, Y_) - var_sum_Y + divergence_sum_Y)

    if hook is not None:
        hook(loss, labels)

    return DSM_loss+lamda*sure_loss
