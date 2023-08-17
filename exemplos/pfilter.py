#Arquivo de funções e classes
import numpy as np
import numpy.ma as ma
print ('hello')

def make_heat_adjusted(sigma):
    def heat_distance(d):
        return np.exp(-d ** 2 / (2.0 * sigma ** 2))

    return heat_distance


## Reamostragem com base no repositório: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
## Autor original é Roger Labbe. O código tem uma licença MIT.
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    return create_indices(positions, weights)


def stratified_resample(weights):
    n = len(weights)
    positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
    return create_indices(positions, weights)


def residual_resample(weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # Reamostragem multinormal para preencher
    residual = weights - num_copies
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


def create_indices(positions, weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### Encerra a reamostragem do rlabbe


def multinomial_resample(weights):
    return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))


# Função de reamostragem de http://scipy-cookbook.readthedocs.io/items/ParticleFilter.html
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices


# Função identidade para esclarecer nomes
identity = lambda x: x


def squared_error(x, y, sigma=1):

    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma ** 2))

#Aplica ruído distribuído no array N,D
def gaussian_noise(x, sigmas):
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    return x + n

#Aplica ruído t ao array x de N,D
def t_noise(x, sigmas, df=1.0):
    n = np.random.standard_t(df, size=(x.shape[0], len(sigmas))) * sigmas
    return x + n

#Aplica ruído Cauchy ao array x de N,D
def cauchy_noise(x, sigmas):
    n = np.random.standard_cauchy(size=(x.shape[0], len(sigmas))) * np.array(sigmas)
    return x + n

#Recebe uma lista de funções que criam n amostras de uma distribuição e junta o resultado em uma matriz N,D
def independent_sample(fn_list):

    def sample_fn(n):
        return np.stack([fn(n) for fn in fn_list]).T

    return sample_fn

#O objeto mantém o estado da população de partículas, podendo ser atualizado a partir das observações
class ParticleFilter(object):

    def __init__(
        self,
        prior_fn,
        observe_fn=None,
        resample_fn=None,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        resample_proportion=None,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=None,
        n_eff_threshold=1.0,
    ):

        self.resample_fn = resample_fn or resample
        self.column_names = column_names
        self.prior_fn = prior_fn
        self.n_particles = n_particles
        self.init_filter()
        self.n_eff_threshold = n_eff_threshold
        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn or identity
        self.dynamics_fn = dynamics_fn or identity
        self.noise_fn = noise_fn or identity
        self.weight_fn = weight_fn or squared_error
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.transform_fn = transform_fn
        self.transformed_particles = None
        self.resample_proportion = resample_proportion or 0.0
        self.internal_weight_fn = internal_weight_fn
        self.original_particles = np.array(self.particles)

	#Inicia o filtro desenhando amostras iniciais
    def init_filter(self, mask=None):
        new_sample = self.prior_fn(self.n_particles)

        # Reamostragem
        if mask is None:
            self.particles = new_sample
        else:
            self.particles[mask, :] = new_sample[mask, :]
	#Atualiza o estado do filtro de partícula usando as obervações
    def update(self, observed=None, **kwargs):
        # Aplicando dinâmicas e ruído
        self.particles = self.noise_fn(
            self.dynamics_fn(self.particles, **kwargs), **kwargs
        )

        # Observações hipotéticas
        self.hypotheses = self.observe_fn(self.particles, **kwargs)

        if observed is not None:
            # Computa as semelhanças junto ás observações

            weights = np.clip(
                self.weights * np.array(
                    self.weight_fn(
                        self.hypotheses.reshape(self.n_particles, -1),
                        observed.reshape(1, -1),
                        **kwargs
                    )
                ),
                0,
                np.inf,
            )
        else:
            # Sem observações, pesos iguais
            weights = self.weights * np.ones((self.n_particles,))

        # Insere peso usando o estado interno
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(
                self.particles, observed, **kwargs
            )
            internal_weights = np.clip(internal_weights, 0, np.inf)
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights

        # Normalizar pesos de acordo com as reamostragens
        self.weight_normalisation = np.sum(weights)
        self.weights = weights / self.weight_normalisation

        # Computar o tamanho efetivo da amostra e o vetor peso
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / np.sum(self.weights ** 2)) / self.n_particles
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))

        # Mantém a amostra atual antes da atualização
        self.original_particles = np.array(self.particles)

        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)
        argmax_weight = np.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]
        self.original_weights = np.array(self.weights) # before any resampling

        # Aplicando pós-processamento
        if self.transform_fn:
            self.transformed_particles = self.transform_fn(
                self.original_particles, self.weights, **kwargs
            )
        else:
            self.transformed_particles = self.original_particles

        # Etapa de reamostragem
        if self.n_eff < self.n_eff_threshold:
            indices = self.resample_fn(self.weights)
            self.particles = self.particles[indices, :]
            self.weights = np.ones(self.n_particles) / self.n_particles

        # Reamostragem de partículas inicias aleatórias
        if self.resample_proportion > 0:
            random_mask = (
                np.random.random(size=(self.n_particles,)) < self.resample_proportion
            )
            self.resampled_particles = random_mask
            self.init_filter(mask=random_mask)
