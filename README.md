<img src="assets/heading.png" style="zoom: 60%;" />

---

**Ada** **P**ath **T**racer is a simple Monte Carlo path tracing renderer based on [Taichi Lang](https://www.taichi-lang.org/), with which you can play easily. The name `AdaPT` is given by my GF and I think this name is brilliant. The icon (first version) is inspired by the *oracle bone script of Chinese character "光"*, which means 'Light'. 

This renderer is implemented based on **MY OWN** understanding of path tracing and other CG knowledge, therefore I **DO NOT** guarantee usability (also, I have done no verification experiments). The output results just... look decent:

##### Steady state rendering

|         "The cornell spheres"          |         "The cornell boxes"         | "Fresnel Blend" |
| :------------------------------------: | :---------------------------------: | :------------------------------------: |
| ![](./assets/adapt-cornell-sphere.png) | ![](./assets/adapt-cornell-box.png) | ![pbr-big-bdpt](https://user-images.githubusercontent.com/126778364/225679926-f75aab9f-0f47-4f45-ab4a-3ea7eaf34055.png)|
|         "The cornell volume box"       |         "BDPT cbox 64 spp"         | "Giant mirror ball" |
| ![pbr-cbox-bdpt](https://user-images.githubusercontent.com/126778364/225680094-8084c378-1533-4b74-871e-4524fff88f28.png)| ![cbox-64-bdpt](https://user-images.githubusercontent.com/46109954/223172423-bec7ac02-8533-432e-9bef-4f02bb4ddbb9.png) | ![pbr-single-ball-bdpt-single-ball](https://user-images.githubusercontent.com/126778364/225680022-ffeb3380-eeab-4beb-9bff-d3c631c36204.png)|

##### Transient state rendering

Note that the gifs presented here are made by compressed jpeg files and optimized (compressed gif). The actual number of images for making the gif is divided by 2, due to the large size of the resulting gif.

|         Transient balls (camera unwarped[^foot])          |         Transient cornell box (camera warped[^foot])         |
| :------------------------------------: | :---------------------------------: |
| ![ezgif-2-7af135f165](https://user-images.githubusercontent.com/126778364/226910459-ee6a3dbd-ad12-480d-a257-8dac1d038842.gif)|![ezgif-4-ab2bd63172](https://user-images.githubusercontent.com/126778364/226910971-3764eb68-9e29-41bd-894d-4a27e9dc49d7.gif)|

[^foot]: 'Camera unwarped' means the transient profile shows the time when a position in the scene is *hit* by emitter ray. 'Camera warped' means the transient profile shows the total time of a position being hit by the emitter ray which should finally transmits to the camera.

Here are the features I currently implemented and supports:

- A direct component renderer: a interactive visualizer for direct illumination visualization
- A **unidirectional / bidirectional Monte-Carlo MIS path tracer**: supports as many bounce times as you wish, and the rendering process is based on Taichi Lang, therefore it can be very fast (not on the first run, the first run of a scene might take a long time due to taichi function inlining, especially for BDPT). The figures displayed above can be rendered within 15-20s (with cuda-backend, GPU supported). The rendering result is displayed incrementally, or maximum iteration number can be pre-set.
- **Volumetric path tracer** that supports uni/bidirectional path tracing in both bounded and unbounded condition
- A **transient renderer** with which you can visualize the propagation of the global radiance.
- Global / indirect illumination & Ability to handle simple caustics
- BRDFs: `Lambertian`, `Modified Phong` (Lafortune and Willems 1994), `Fresnel Blend` (Ashikhmin and Shirley 2002), `Blinn-Phong`, `Mirror-specular`.
- BSDFs (with medium): deterministic refractive (glass-like)
- mitusba-like XML scene file definition, supports mesh (from wavefront `.obj` file) and analytical sphere.
- scene visualizer: which visualizes the scene you are going to render, helping to set some parameters like the relative position and camera pose
- Extremely easy to use and multi-platform / backend (thanks to Taichi), with detailed comments and a passionate maintainer (yes, I myself). Therefore you can play with it with almost no cost (like compiling, environment settings blahblahblah...)

BTW, I am just a starter in CG (ray-tracing stuffs) and Taichi Lang, so there WILL BE BUGS or some design that's not reasonable inside of my code. Also, I haven't review and done extensive profiling & optimization of the code, therefore again --- correctness is not guaranteed! But, feel free to send issue / pull-request to me if you are interested.

If you are indeed interested and you want to run a trial... try it yourself first and ask for help if you failed. I just think writing documents is a pain-in-the-ass...

For the commit logs, please refer to another repo: [Enigmatisms/learn_taichi](https://github.com/Enigmatisms/learn_taichi). This repo is isolated from my Taichi Lang learning repo.

---

AdaPT is licensed under GPL-v3.
