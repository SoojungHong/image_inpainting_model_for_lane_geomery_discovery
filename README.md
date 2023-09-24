## Correcting Faulty Road Maps by image in painting

As maintaining road networks is labor-intensive, many au-
tomatic road extraction approaches have been introduced to
solve this real-world problem, fueled by the abundance of
large-scale high-resolution satellite imagery and advances in
computer vision. However, their performance is limited for
fully automating the road map extraction in real-world ser-
vices. Hence, many services employ the two-step human-in-
the-loop system to post-process the extracted road maps: er-
ror localization and automatic mending for faulty road maps.
Our paper exclusively focuses on the latter step, introduc-
ing a novel image inpainting approach for fixing road maps
with complex road geometries without custom-made heuris-
tics, yielding a method that is readily applicable to any road
geometry extraction model. We demonstrate the effectiveness
of our method on various real-world road geometries, such as
straight and curvy roads, T-junctions, and intersections.

## Globally Locally Consistent Roadmap Correction (CLCRC) model architecture 

![plot](./figures/extended_GLCIC_architecture.png)


## Inpainting results on faulty road segments 
<img src="https://github.com/SoojungHong/image_inpainting_model_for_lane_geomery_discovery/assets/17648100/4ae43d85-ee43-479a-b2d3-e212d55482dc" width="500" />

## Inpainting results comparison 

We conduct quantitative and qualitative analysis on whether the modifications are effective. We
use three metrics, Correctness, Completeness, and Quality,
which are widely used in road extraction tasks We choose four random road maps per
each road type: Straight, Curvy, T-junction, and Intersection,
where the latter three types are known to be challenging for
existing post-processing methods. 
Table 1 shows the performance impact on various modi-
fications: GLCRC being the architectural and GLCRC+L be-
ing the training loss modifications. We can clearly observe
that our approach shows the best image quality in the road ge-
ometry inpainting problem. Furthermore, Figure 1 shows the
effectiveness of our method on various road types, implying that it understands underlying road geometries to reconstruct
the road location.
<img src="https://github.com/SoojungHong/image_inpainting_model_for_lane_geomery_discovery/assets/17648100/dd4e2c27-acc4-4b70-a5ab-dad8a87417b2" width="400" />
