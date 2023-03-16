<img src="assets/heading.png" style="zoom: 60%;" />

---

**Ada** **P**ath **T**racer is a simple Monte Carlo path tracing renderer based on [Taichi Lang](https://www.taichi-lang.org/), with which you can play easily. The name `AdaPT` is given by my GF and I think this name is brilliant. The icon (first version) is inspired by the *oracle bone script of Chinese character "光"*, which means 'Light'. 

This renderer is implemented based on **MY OWN** understanding of path tracing and other CG knowledge, therefore I **DO NOT** guarantee usability (also, I have done no verification experiments). The output results just... look decent:

|         "The cornell spheres"          |         "The cornell boxes"         | "Fresnel Blend" |
| :------------------------------------: | :---------------------------------: | :------------------------------------: |
| ![](./assets/adapt-cornell-sphere.png) | ![](./assets/adapt-cornell-box.png) | ![pbr-big-bdpt](https://user-images.githubusercontent.com/126778364/225679926-f75aab9f-0f47-4f45-ab4a-3ea7eaf34055.png)|
|         "The cornell volume box"       |         "BDPT cbox 64 spp"         | "Giant mirror ball" |
| ![pbr-cbox-bdpt](https://user-images.githubusercontent.com/126778364/225680094-8084c378-1533-4b74-871e-4524fff88f28.png)| ![cbox-64-bdpt](https://user-images.githubusercontent.com/46109954/223172423-bec7ac02-8533-432e-9bef-4f02bb4ddbb9.png) | ![pbr-single-ball-bdpt-single-ball](https://user-images.githubusercontent.com/126778364/225680022-ffeb3380-eeab-4beb-9bff-d3c631c36204.png)|



Here are the features I currently implemented and supports:

- A direct component renderer: a interactive visualizer for direct illumination visualization
- A unidirectional / bidirectional Monte-Carlo MIS path tracer: supports as many bounce times as you wish, and the rendering process is based on Taichi Lang, therefore it can be very fast (not on the first run, the first run of a scene might take a long time due to taichi function inlining, especially for BDPT). The figures displayed above can be rendered within 15-20s (with cuda-backend, GPU supported). The rendering result is displayed incrementally, or maximum iteration number can be pre-set.
- Volumetric path tracer that supports uni/bidirectional path tracing in both bounded and unbounded condition
- Global / indirect illumination & Ability to handle simple caustics
- BRDFs: `Lambertian`, `Modified Phong` (Lafortune and Willems 1994), `Fresnel Blend` (Ashikhmin and Shirley 2002), `Blinn-Phong`, `Mirror-specular`.
- BSDFs (with medium): deterministic refractive (glass-like)
- mitusba-like XML scene file definition, supports mesh (from wavefront `.obj` file) and analytical sphere.
- scene visualizer: which visualizes the scene you are going to render, helping to set some parameters like the relative position and camera pose
- Extremely easy to use, with detailed comments and a passionate maintainer (yes, I myself). Therefore you can play with it with almost no cost (like compiling, environment settings blahblahblah...)

BTW, I am just a starter in CG (ray-tracing stuffs) and Taichi Lang, so there WILL BE BUGS or some design that's not reasonable inside of my code. Also, I haven't review and done extensive profiling & optimization of the code, therefore again --- correctness is not guaranteed! But, feel free to send issue / pull-request to me if you are interested.

If you are indeed interested and you want to run a trial... try it yourself first and ask for help if you failed. I just think writing documents is a pain-in-the-ass...

For the commit logs, please refer to another repo: [Enigmatisms/learn_taichi](https://github.com/Enigmatisms/learn_taichi). This repo is isolated from my Taichi Lang learning repo.

---

AdaPT is licensed under GPL-v3.
