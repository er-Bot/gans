# gans

A list of different gan architectures.

# Content

- [GAN](https://arxiv.org/abs/1406.2661) 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\large&space;\min_G&space;\max_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{\text{data}}(x)}[\log&space;D(x)]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p_{z}(z)}[\log&space;(1&space;-&space;D(G(z)))]." target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{120}&space;\large&space;\min_G&space;\max_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{\text{data}}(x)}[\log&space;D(x)]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p_{z}(z)}[\log&space;(1&space;-&space;D(G(z)))]." title="\large \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z)))]." /></a>

![](img/cgan.png)

- [CGAN]()https://arxiv.org/abs/1411.1784 : 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{120}&space;\large&space;\min_G&space;\max_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{\text{data}}(x)}[\log&space;D(x&space;|&space;y)]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p_{z}(z)}[\log&space;(1&space;-&space;D(G(z&space;|&space;y)&space;|&space;y))]." target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\dpi{120}&space;\large&space;\min_G&space;\max_D&space;V(D,&space;G)&space;=&space;\mathbb{E}_{x&space;\sim&space;p_{\text{data}}(x)}[\log&space;D(x&space;|&space;y)]&space;&plus;&space;\mathbb{E}_{z&space;\sim&space;p_{z}(z)}[\log&space;(1&space;-&space;D(G(z&space;|&space;y)&space;|&space;y))]." title="\large \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x | y)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z | y) | y))]." /></a>
