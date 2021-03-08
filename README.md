# gans

A list of different gan architectures.

# Content

- [GAN](https://arxiv.org/abs/1406.2661) 
\begin{equation}
\min_G \max_D V(D, G) = \mathbb{E}_{\bm{x} \sim p_{\text{data}}(\bm{x})}[\log D(\bm{x})] + \mathbb{E}_{\bm{z} \sim p_{\bm{z}}(\bm{z})}[\log (1 - D(G(\bm{z})))].
\end{equation}

- [CGAN]()https://arxiv.org/abs/1411.1784 : 
\begin{equation}
\min_G \max_D V(D, G) = \mathbb{E}_{\bm{x} \sim p_{\text{data}}(\bm{x})}[\log D(\bm{x} | \bm{y})] + \mathbb{E}_{\bm{z} \sim p_{\bm{z}}(\bm{z})}[\log (1 - D(G(\bm{z} | \bm{y}) | \bm{y}))].
\end{equation}
