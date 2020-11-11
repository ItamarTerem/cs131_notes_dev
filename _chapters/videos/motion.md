---
title: Motion
keywords: (insert comma-separated keywords here)
order: 15 # Lecture number for 2020
---

# Table of Contents
**Outlined content of Lecture 15 CS 131 Fall 2020

1. 15.1: Optical Flow
    - At a high level, optical flow is the observation of moving images as a function of location and time. The notes on this component of the lecture outline not only the potential applications of motion based data, such as medical imaging, but a concise derivation of the optical flow equation and an outline of the aperature problem.
2. 15.2: Lukas-Kanade Method
    - The Lukas-Kanade Method is one method of determining the direction of optical vectors. More specifically, this approach applies the spatial coherence constraint, which assumes that neighboring pixels in an image have the same value for optical flow, which is why this method is usually used for calculations in which the change in pixels between frames is relatively small.
3. 15.3: Pyramids for Large Motion
    - Whereas the Lukas-Kanade Method for optical flow assumes a small change in pixels between frames, this model can fail dramatically when applied to larger moving systems of pixels. The Pyramids for Large Motion method applies a similar methodology to the sliding window method of image analysis in that it generates a pyramid of images for the purposes of 
4. 15.4: Horn-Schunk Method
5. 15.5: Motion Segmentation
6. 15.6: Motion: Applications

# Optical Flow
**From Images To Videos** \
-- Video is a sequence of frames capture over time. As a result, the image now is a function of the location and time, namely $I(x,y,t)$.

-- Optical flow (from Wikipedia) is the pattern of apparent motion of objects, surfaces, and edges in a visual scene caused by the relative motion between an observer and a scene. 

It is important to note that apparent motion can be cause by lighting changes without actual motion. We can think of a rotating sphere under fix lighting versus a stationary sphere under moving illumination. 

**Our goal:** is to recover image motion at each pixel from optical flow.


**Why is motion useful?**

There are many applications where motion is a useful information. One application is in medical imaging. The figure below shows the optical flow motion of human brain (sagittal coronal and axial planes) during the cardiac cycle. The motion is due to blood pulsation and cerebrospinal fluid (CSF) flow. Abnormal motion can indicate pathologist conditions like Chiari I Malformation. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6269230/) 

![](https://i.imgur.com/0GIJOql.gif)
Brain in Motion © 2020 Itamar Terem

**Estimating Optical Flow**

Given two subsequent frames, we would like to estimate the apparent motion field $u(x,y)$ and $v(x,y)$.

In order to derive the equation we will assume three key assumptions:

1. Brightness constancy - projection of the same point looks the same in every frame. 
2. Small motion - points do not move very far (exhibit small motion). This means that the image motion of a surface patch chnages gradually over time. 
3. Spatial coherence: neighboring points in the scene will have the similar motion. 



The relationship between a frame at time $t$ and at time $t+ \Delta t$ can be written as follow (assumption 1):

$I(x,y,t) = I(x+u,y+v,t+ \Delta t)$

Under assumption 2 the right hand side can be approximate by the first two terms in the Taylor series:


$I(x,y,t) = I(x+u,y+v,t+ \Delta t) \approx I(x,y,t) + I_{x}*u + I_{y}*v +I_{t}* \Delta t$ 

Where $I_{x}$ is the image derivative along $x$, $I_{y}$ is the image derivative along $y$, abd $I_{t}$ is the image derivative along $t$. 

This will result in:

$I_{x}*u + I_{y}*v +I_{t}* \Delta t \approx 0$

We can set $\Delta t = 1$ (velocity estimation can be achieved by taking $\Delta t$ into account), and write the equation in a vectorize form: 

$\nabla I*[u,v]^T +I_{t} = 0$

This last equation is called the optical flow constraint equation since it expresses a constraint on the components $u$ and $v$ of the optical flow. This is an equation in two unknowns and cannot be solved as such. This is known as the aperture problem of the optical flow algorithms. To find the optical flow another set of equations is needed, given by some additional constraint. All optical flow methods introduce additional conditions for estimating the actual flow. On of these constrain is the spatial coherence assumption. Two if these methods, the Lukas-Kanade and Horn-Schunk, will be introduced in the following sections.


**The aperture problem**

When looking at the gradient part in the equation,  $\nabla I*[u,v]^T$ , we can see that we cannot determine the component of the optical flow at $\pi/2$ to the gradient direction (i.e., parallel to the edge). This ambiguity is known as the aperture problem.

We can show that as follow:

- If $(u,v)$ satisfies the equation then $\nabla I*[u,v]^T +I_{t} = 0$
- Let's assume $(u',v')$ is perpendicular to $\nabla I$, then $\nabla I*[u',v']^T = 0$

Therefore: 

$\nabla I*[u+u',v+v']^T +I_{t} = 0$ ,  which means that the point $(u+u',v+v')$ also satisfies the equation. An illustration can be seen in the figure below.

![](https://i.imgur.com/k0wy0Ac.jpg)

*(scource: Silvio Savarese) *


The fact that our image is captured by  a finite aperture limits our abillity to capture the true motion. And as we can see, for any given vector $(u,v)$ the component we can actually measure is the one that perpendicular to the edge.

These can be seen in the Barberpole illusion. The The apparent motion through the small aperture is  downward and to the right, when in fact the real motion is, which can be seen trough the large aperture is downward.


![](https://upload.wikimedia.org/wikipedia/commons/f/f0/Aperture_problem_animated.gif)

![](https://upload.wikimedia.org/wikipedia/commons/3/39/Barberpole_illusion_animated.gif)


Summary:

- We defined the optical flow problem, and derive it's equation under the key assumtions.
- We defined the aperture problem and saw that the optical flow constraint equation has two unknowns and cannot be solved without some additional constraint,

In the next sections we will learn about methods to solve the optical flow equation like the Lukas-Kanade and Horn-Schunk.




# Lukas - Kanade Method

In the previous section we defined the problem of optical flow and derived equations to help us esitmate the motion vectors of individual pixels in an image. Now we propose three methods for solving these equations. We will first introduce the Lukas-Kanade method. 

**Approach**

With only one equation and two unknows, it is impossible to estimate the image motion given by $(u, v)$ since solving for $u$ and $v$ requires at least two equations which means that we need more equations to solve for the unknowns. 

To derive a system of equations, the Lukas-Kanade method applies the **Spatial coherence constraint** (spatial coherence introduced in the previous section) to the system; this assumes that neighboring pixels have the same optical flow value, $(u, v)$. Applying this constraint to a $5 \times 5$ window in the image yields 25 equations per pixel as follows:

\begin{equation*}
\nabla I(p_i) \cdot [u,v]^T  +I_{t}(p_i) = 0
\end{equation*}

<center>

![](https://i.imgur.com/4VSIDYn.png)

</center>




The above overly-constrained  system of equations can be rewritten in the form $Ad = b$. To solve for $d$, a least squares method for solving overly-constrained system of equations is applied. The solution is given by $(A^TA)d = A^Tb$ and can explicitly be written in the form:

<center>

![](https://i.imgur.com/mM8308X.png)

</center>


**When is Lukas-Kanade equation Solvable?**

The Lukas-Kanade system of equations is solvable when the following conditions hold, namely:

* $A^TA$ should be invertible.
* $A^TA$ should not be too small to avoid issues with noise i.e eigenvalues $\lambda_{1}$ and  $\lambda_{2}$ should not be too small.
* $A^TA$ should be well-conditioned i.e eigenvalues $\lambda_{1}/ \lambda_{2}$ should not be too large when $\lambda_{1} > \lambda_{2}$

**Intepretations of Above Conditions**

From the above conditions, we can infer that $A^TA$ is the same as the second moment matrix of the Harris corner detector:

<center>

![](https://i.imgur.com/W4l5kao.png)

</center>


This implies that solving the Lucas-Kanade equations using the matrix $A^TA$ can also be interpreted as trying to track corners detected by a harris corner detector in an image. As such the  larger eigenvector and eigenvalues of $A^TA$ can be associated with the direction of fastest change in intensity (an edge) and the magnitude of this change respectively with the smaller eigenvector othorgonal to the larger one. We can also intepret the eigenvalues of $A^TA$ using the following geometric representation.


![](https://i.imgur.com/try9hL3.png)



It is apparent from the coditions under which the Lukas-Kanade equations are solvable that the best eigenvalues lie in the blue region around a corner where eigengvalues $\lambda_{1}$ and $\lambda_{2}$ are both large. If we chose values that are too small then that falls in the flat region. If chose eigenvalues in the orange regions then we would be in an edge and optical flow suffers from aperture problems around edges. These choices are better visualized using an edge, low-texture and high-texture regions represented in the image below.

![](https://i.imgur.com/PTgNC2v.png)

In the first image the selected region contains an edge where gradients maybe very large or very small (either $\lambda_{1}$ large or $\lambda_{2}$ large). In the middle image, the selected region is a low-texture flat region where gradients are small ($\lambda_{1}$ small and $\lambda_{2}$ small ) and in the last image, the selected region is a high-texture region where gradients are high ($\lambda_{1}$ large and $\lambda_{2}$ large ) and this would be the preferred region  

**Improving Accuracy of Optical Flow Estimation**

We can improve the accuracy of Lukas-Kanade by reintroducing previously dropped higher order terms from the Taylor exanpsion approximation back into the equation. The equation 

<center>

![](https://i.imgur.com/6PirtyT.png)

</center>


can then be rewritten as:

<center>

![](https://i.imgur.com/Q8hcMvI.png)

</center>
The equation is a polynomial root finding problem that can be solved by Newton's method. Many iterations of newton's method can lead to better results.

We can also use the method of iterative refinement to improve the accuracy of optical flow estimation. The Lukas-Kanade iterative algorithm can be described as follows:

* Estimate velocity at each pixel by solving Lucas-Kanade equations.
* Warp $I(t-1)$ towards $I(t)$ using the estimated flow field and image warping techniques.
* Repeat until convergence.

**Errors in Lukas-Kanade**

The Lukas-Kanade method is used under the constraint of assumptions of optical flow and these means that if some of these assumptions are violated the approach may not work accurately. For example;

* When brightness consistency doesn't hold and pixels change intensity. 
* When a point doesn't move like it's neighbors 
* When the motion is not small.

In the next section we improve on the Lukas-Kanade to mitigate some of the errors outlined above.

# Pyramids For Large Motion
Recall that, when implementing the Lukas-Kanade method, we assume that the change in position of pixels between frames is relatively small (usually the size of one pixel or less). However, if the movement across the image is large, the Lukas-Kanade method can fail dramatically. The solution is to use the Lukas-Kanade method on downsampled versions of the original video.

**Lucas-Kanade Without Pyramids**
![](https://i.imgur.com/Y9SuZS6.png)

As you can see, using the Lucas-Kanade without preprocessing leads to inconsistent movement vectors that define the movement of the tree. The Lucas-Kanade method has a strong bias towards the inital positions of the tree, which is difficult to respect while also trying to capture large motion away from the inital positions. 

**Lukas-Kanade With Pyramids**

To correct for this problem, we use the same strategy we used in our sliding-window feature detector: we create an image-pyramid of the original images and run Lucas-Kanade iteratively over each level of the pyramid.

![](https://i.imgur.com/3MI4pJP.png)

This way, movement that is expressed over multiple pixels in the original image might be expressed over a single pixel in one of the lower resolution images, allowing Lucas-Kanade to be more effective by fulfilling the assumption that motion across the image is small.

![](https://i.imgur.com/qc22W1w.png)

As you can see, the Lucas-Kanade method now has many more consistent vectors defining the movement of the tree across the image.


# Horn - Schunk Method - Ryan Samadi & Oscar O'Rahilly

The Horn-Schunk method is another way to compute optical flow. The central idea behind this method is through formulating flow as the following global energy function. 

\begin{equation*}
E=\int\int[(I_xu+I_yv+I_t)^2+\alpha^2(||\nabla u||^2+||\nabla v||^2)]\mathrm{d}x \mathrm{d}y
\end{equation*}

Our goal is to minimize the energy function with respect to $u(x, y)$ and $v(x,y)$.

To understand the process of minimization, it is very helpful to recognize the different parts of our energy function. 

The first part, highlighted in red, reflects the brightness constancy assumption, which states that the projection of the same point looks the same in every frame.

![](https://i.imgur.com/8FYqMeb.png)

According to this assumption, $I_xu+I_yv+I_t$ should be $0$. Note that we square this expression to ensure that the value of the brightness constancy is as close to $0$ as possible.

The second term in the energy function is the smoothness constraint. It reflects the small motion assumption, which states that every point moves only by small amounts between successive frames. Note that we square the magnitudes of $u$ and $v$ to encourage smoother flow as we are trying to make sure that only small changes are made to the position of each point.

![](https://i.imgur.com/3VRr3Fv.png)


The third and final term in our energy function is the regularization constant, $\alpha$. Note that larger values of $\alpha$ lead to a smoother flow.

![](https://i.imgur.com/CfMdD9c.png)

**Minimizing the energy function**

The minimization task can be solved by taking the derivative with respect to $u$ and $v$ and setting them equal to zero. This yields the following two equations:

\begin{equation*}
I_x(I_xu+I_yv+I_t)-\alpha^2\Delta u=0 \\
I_y(I_xu+I_yv+I_t)-\alpha^2\Delta v=0
\end{equation*}

Notice that in these equations we have the **lagrange operator**, $\Delta=\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}$, which is related to the second derivative with respect to $x$ and $y$. In practise it is computed as 

\begin{equation*}
\Delta u(x,y)=\bar{u}(x,y)-u(x,y)\
\end{equation*}

where $\bar{u}(x,y)$ is the weighted average of $u$ in a neighborhood around $(x,y)$. Substituting this expression into the two equations above yields the following

\begin{equation*}
u^{k+1}=\bar{u}^k-\frac{I_x(I_x\bar{u}^k+I_y\bar{v}^k+I_t)}{\alpha^2+I_x^2+I_y^2} \\
v^{k+1}=\bar{v}^k-\frac{I_y(I_x\bar{u}^k+I_y\bar{v}^k+I_t)}{\alpha^2+I_x^2+I_y^2}
\end{equation*}

Now we have a linear equation in $u$ and $v$ for each pixel. 

**Iterative Horn-Schunk**

Since the solution depends on the neighboring values of the optical flow field, we must recalculate the values of $u$ and $v$ iteratively once the neighbors have been updated. This can be done as follows

\begin{equation*}
u^{k+1}=\bar{u}^k-\frac{I_x(I_x\bar{u}^k+I_y\bar{v}^k+I_t)}{\alpha^2+I_x^2+I_y^2} \\
v^{k+1}=\bar{v}^k-\frac{I_y(I_x\bar{u}^k+I_y\bar{v}^k+I_t)}{\alpha^2+I_x^2+I_y^2}
\end{equation*}

We calculate this by first assuming an initial value of the average optical flow, $\bar{u}^0$ and $\bar{v}^0$, and use this to calculate subsequent values of the optical flow.

**What does the smoothness regulatization do?**

The smoothness regularization term $||\nabla u||^2+||\nabla v||^2$, is a sum of squared terms which we are putting into the expression to be minimized.

Note that in texture free regions, there is no optical flow and on edges meaning that points will flow to the nearest points, solving the aperture problem.

![](https://i.imgur.com/z6TsyaS.png)

**Dense Optical Flow with Michael Black's method**

Michael Black, took this idea further. He replaced the regularization term (which is a quadratic function)

![](https://i.imgur.com/YD40z0r.png)

and replaced it with the following function.

![](https://i.imgur.com/KvwEJYn.png)

This regularization term works better as it assigns a smaller penalty of the magnitude of $u$ for larger values of $u$ than the quadratic term. 


# Motion Segmentation 
Motion segmentation involves breaking down a picture not by the color of the pixels but such that each segment has a consistent and coherent motion.

![](https://i.imgur.com/fOJQRrp.jpg)
*(source: Haque, Nazrul & N, Dinesh reddy & Krishna, Madhava. (2017). Joint Semantic and Motion Segmentation for dynamic scenes using Deep Convolutional Networks.)*

**Identifying Layers**

In order to identify these layers let us first see how we can parametrize motion.
We know that the horizontal and vertical componenets of affine motion can be represented as follows: 

$u(x,y) = a_1 + a_2x + a_3y$\
$v(x,y) = a_4 + a_5x + a_6y$

This assumes that all pixels in one segment move the same way.

By substituting the above parametric equations into the brightness consistency equation, we obtain:

$I_x(a_1 + a_2x + a_3y) + I_y(a_4 + a_5x + a_6y) + I_t ≈ 0$,

where $I_x$, $I_y$ and $I_t$ are the gradients of the image with respect to the x-direction, y-direction and time, respectively.

Once again, the assumption is that all pixels in one segment share the same parameters $a_1 - a_6,$ so every pixel provides a linear equation with 6 unknowns.

We can use *least squares minimization* to calculate the parameters of the motion within the segment. 

We need to minimize the following:

$Err(a) = \sum [I_x(a_1 + a_2x + a_3y) + I_y(a_4 + a_5x + a_6y) + I_t]^2$

![](https://i.imgur.com/E6sucJE.png)
*(source: Silvio Savarese)*

**Minimizing Error**

To do this, we map out parameter vectors, $a_i$, into motion parameter space and perform k-means clustering on the affine motion parameter vectors.

The resulting centers of k-means are the values of the parameters $a_1 - a_6$ that minimize the error function. The vectors $a_i$ correspond to the blocks that should be grouped together in one layer. Thus, we have an initial hypothesis of motion segmentation.

Finally, we must assign each pixel to the best estimated hypothesis, perform regional filtering to ensure pixels are connected to each other and finally re-estimate motion within each region.
![](https://i.imgur.com/Q3Yu2wK.png)
*(source: Silvio Savarese)*


# Motion: Applications 

Once optical flow is estimated, there are a lot of useful applications with the motion features. We will cover some examples here. 

**Tracking object**

Object tracking is the most important and wide application of optical flow. One of the major computer vision library, OpenCV, offers the following example to track some points in a video. First, several key points are extracted by applying cv2.goodFeaturesToTrack() function on the first frame of the video. Then, hLucas-Kanade optical flow is iteratively applied on these points with the help of cv2.calcOpticalFlowPyrLK() function. It returns next points along with some status numbers which has a value of 1 if next point is found, else zero. We iteratively pass these next points as previous points in next step. The result is as follows:

![](https://i.imgur.com/6WV7c4g.jpg)

*[Source Here](https://opencv-python-tutroals.readthedocs.io/en/latest/pytutorials/pyvideo/pylucaskanade/pylucaskanade.html#lucas-kanade-optical-flow-in-opencv)*

On the image above, different keypoints are plotted with different colors. These lines represent their paths during the time frame of the video. The next two animations illustrate the full picture of the object tracking application using the object flow. The image on the left is the result using Sparse Optical Flow, which tracks a few key feature points, while the image on the right shows the result using Dense Optical Flow, which estimates the flow of all pixels in the image. 

![](https://i.imgur.com/XdKdLFu.gif)

*[Source Here](https://nanonets.com/blog/optical-flow/).*

There are a lot researches and applications on traffic analysis with the help of optical flow. As more and more efforts are spent on the self-driving car, building an intellgient traffic system is a critical part to achieve L5 self-driving. 

**Segmenting objects based on motion cues**
* Background Substraction

Assuming there is a staitic camera observing a scene, the goal is to separate the static background from the moving foreground. We can calucate the optical flow and decide which pixels never move. These pixels are highly likely to be the background of the scene. This application is widely used on surveillance camera system, as shown in the following figure, (scource: Silvio Savarese)

![](https://i.imgur.com/Yke3xn5.png)

*(scource: Silvio Savarese)
*

POSTECH Computer Vision Lab has performed a lot of reseraches on this topic. Following is a video to showcase the application of Generalized Background Subtraction, which was published on ICCV 2011. 

[Generalized Background Subtraction](https://www.youtube.com/watch?v=YAszeOaInUM)


* Motion segmentation
In this application, the pixels are divided into multiple coherently moving objects, so that we can do grouping and clustering. On the following example, we can see the Group A, B C have different motions, and thus resulting in different segments. 

![](https://i.imgur.com/ES2m67g.png)
*Source: S.J. Pundlik and S.T. Birchfield, Motion Segmentation at Any Speed, Proceedings of the British Machine Vision Conference (BMVC) 2006
*

Here is the [Youtube video](https://www.youtube.com/watch?v=ubQGZ3DAUAA) demonstrating the application on a surveillance video on a railway station. 


**Recongnizing events and acitivities**
we can estimate a person's body position. Then by analyzing how the body moves using optical flow, we can recongnize the type of activity the person is performing.  
![](https://i.imgur.com/rzfemHM.png)

*(scource: Zisserman.D. Ramanan, Forsyth, and A. Zisserman. Tracking People by Learning their Appearance, PAMI 2007). 
*

Following example shows different type of spins in the skating.  .

![](https://i.imgur.com/Z6m5bNf.png)

*(scource: Juan Carlos Niebles, Hongcheng Wang and Li Fei-Fei, Unsupervised Learning of Human Action Categories Using Spatial-Temporal Words, BMVC, Edinburgh, 2006)
*

Another example shows the application on a group of peoples, detecting whether people are dancing, talking, jogging and ect. 

![](https://i.imgur.com/rMdTbiy.png)

*(scource: W.Choi & K. Shahid S. Savarese WMC 2010)*


The lectuer of this class Mr. Juan Carlos Niebles, is an active scholoar on this filed. His talk on [Human Event Understanding: From Actions to Tasks](https://tv.vera.com.uy/video/55276) can be found here.  


**Estimating 3D structure**
Given a sequence of frames, we can calucate how pixels move between frames, which helps us to determine 3D structure scene. 

![](https://i.imgur.com/JqOLRQK.png)

*(scource: Silvio Savarese)*
