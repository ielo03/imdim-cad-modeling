//
// Simple OpenSCAD test model for multi-angle rendering
// Features: cube, cylinder, sphere, difference(), translate(), hull()
//

$fn = 64;   // smoother curves

// Base cube
cube([30, 30, 10], center = true);

// Cylinder on top
translate([0, 0, 10])
    cylinder(h = 20, r = 10, center = false);

// Sphere cutout
difference() {
    translate([0, 0, 15])
        sphere(r = 12);

    // Remove bottom half of the sphere so the cutout is visible
    translate([-20, -20, -5])
        cube([40, 40, 20]);
}

// Little rounded nub using hull
hull() {
    translate([15, 15, 0]) sphere(r = 4);
    translate([15, 15, 8]) sphere(r = 2);
}