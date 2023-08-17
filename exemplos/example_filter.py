#Alunos: Bruno Machado Ferreira(181276), Ernani Neto(180914), Fábio Gomes(181274) e Ryan Nantes(180901)
#Filtro de partículas(Código principal)
from pfilter import (
    ParticleFilter,
    gaussian_noise,
    cauchy_noise,
    t_noise,
    squared_error,
    independent_sample,
)
import numpy as np

# testing only
from scipy.stats import norm, gamma, uniform
import skimage.draw
from skimage.draw import (line, polygon, circle_perimeter,
                          ellipse, ellipse_perimeter,bezier_curve)

import cv2


img_size = 100


def blob(x):
    """Dada uma matriz de 3 colunas, com as posicoes e tamanho do blob, 
    cria uma imagens de dimensões img_size x img_size, com blobs inseridos
    a partir do valor das fileiras de x
    
    One row of x = [x,y,radius]."""
    y = np.zeros((x.shape[0], img_size, img_size))
    for i, particle in enumerate(x):
        rr, cc = skimage.draw.circle(
            particle[0], particle[1], max(particle[2], 1), shape=(img_size, img_size)
        )
        y[i, rr, cc] = 1
    return y

columns = ["x", "y", "radius", "dx", "dy"]


# Amostragem das varáveis
# (x e y sendo coordenadas com valor máximo do tamanho da imagem)
prior_fn = independent_sample(
    [
        norm(loc=img_size / 2, scale=img_size / 2).rvs,
        norm(loc=img_size / 2, scale=img_size / 2).rvs,
        gamma(a=1, loc=0, scale=10).rvs,
        norm(loc=0, scale=0.5).rvs,
        norm(loc=0, scale=0.5).rvs,
    ]
)

# Ajusta a velocidade das partículas
def velocity(x):
    dt = 1.1
    xp = (
        x
        @ np.array(
            [
                [1, 0, 0, dt, 0],
                [0, 1, 0, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ).T
    )

    return xp


def example_filter():
    # Criando o filtro
    pf = ParticleFilter(
        prior_fn=prior_fn,
        observe_fn=blob,
        n_particles=200,
        dynamics_fn=velocity,
        noise_fn=lambda x: t_noise(x, sigmas=[0.15, 0.15, 0.05, 0.05, 0.15], df=120.0),
        weight_fn=lambda x, y: squared_error(x, y, sigma=2),
        resample_proportion=0.1,
        column_names=columns,
    )

    # np.random.seed(2018)
    # Iniciando no centro da tela
    s = np.random.uniform(10, 5)

    # Movimentação aleatória
    dx = np.random.uniform(-0.25, 0.25)
    dy = np.random.uniform(-0.25, 0.25)

    # No centro
    x = img_size // 2
    y = img_size // 2
    scale_factor = 30

    # Criando UI
    cv2.namedWindow("Resultados", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resultados", scale_factor * img_size, scale_factor * img_size)

    for i in range(200):
        # Gerando a imagem
        low_res_img = blob(np.array([[x, y, s]]))
        pf.update(low_res_img)

        # Ajustar escala da imagem
        img = cv2.resize(
            np.squeeze(low_res_img), (0, 0), fx=scale_factor, fy=scale_factor
        )

        cv2.putText(
            img,
            "ESC para fechar",
            (100, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (255, 255, 255),
            6,
            cv2.LINE_AA,
        )

        color = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)

        x_hat, y_hat, s_hat, dx_hat, dy_hat = pf.mean_state

        # Inserir as partículas
        for particle in pf.original_particles:

            xa, ya, sa, _, _ = particle
            sa = np.clip(sa, 1, 100)
            cv2.circle(
                color,
                (int(ya * scale_factor), int(xa * scale_factor)),
                max(int(sa * scale_factor), 1),
                (1, 0, 0),
                1,
            )

        cv2.circle(
            color,
            (int(y_hat * scale_factor), int(x_hat * scale_factor)),
            max(int(sa * scale_factor), 1),
            (0, 1, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            color,
            (int(y_hat * scale_factor), int(x_hat * scale_factor)),
            (
                int(y_hat * scale_factor + 5 * dy_hat * scale_factor),
                int(x_hat * scale_factor + 5 * dx_hat * scale_factor),
            ),
            (0, 0, 1),
            lineType=cv2.LINE_AA,
        )

        cv2.imshow("Resultados", color)
        result = cv2.waitKey(20)
        # Encerra com um botão
        if result == 27:
            break
        x += dx
        y += dy

    cv2.destroyAllWindows()


if __name__ == "__main__":
    example_filter()

# %%
