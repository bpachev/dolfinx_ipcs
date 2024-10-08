SetFactory("OpenCASCADE");
lc = 0.0006;
Box(1) = {0, 0, 0, 2.5, 0.41, 0.41};
Cylinder(2) = {0.5,0.2,-1, 0.0,0.0,20, 0.05};

BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
Mesh.ElementOrder = 1;
Field[1] = Distance;
Field[1].FacesList = {7};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc;
Field[2].LcMax = 15*lc;
Field[2].DistMin = 0.025;
Field[2].DistMax = 1;
Field[7] = Min;
Field[7].FieldsList = {2};
Background Field = 7;

Physical Volume("Fluid", 1) = {3};
Physical Surface("Inlet", 2) = {1};
Physical Surface("Outlet", 3) = {6};
Physical Surface("Walls", 4) = {5, 3, 2, 4};
Physical Surface("Obstacle", 5) = {7};
