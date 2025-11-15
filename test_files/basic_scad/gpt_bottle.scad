// BlenderBottle-style shaker approximation
$fn = 96;

// ------------ Top-level ------------
bottle();

module bottle() {
    // main white body
    color([0.95, 0.95, 0.95])
        bottle_body();

    // gray lid + spout + handle
    color([0.8, 0.8, 0.8])
        translate([0, 0, BODY_H])
            lid_assembly();
}

// ------------ Parameters ------------
BODY_R      = 34;   // body radius (mm-ish)
BODY_H      = 190;  // height of main body
BOTTOM_ROUND= 6;    // bottom fillet radius
LID_H       = 20;   // height of gray lid ring
LID_OVER    = 2;    // lid overhang beyond body
SPOUT_R     = 10;   // spout radius
SPOUT_H     = 18;   // spout height above lid
HANDLE_TH   = 4;    // handle thickness
HANDLE_W    = 22;   // handle width
HANDLE_L    = 55;   // handle length

// ------------ Body ------------
module bottle_body() {
    // main straight section (shortened by bottom round)
    difference() {
        union() {
            // straight wall
            translate([0,0,BOTTOM_ROUND])
                cylinder(h = BODY_H - BOTTOM_ROUND, r = BODY_R);

            // fake rounded bottom using minkowski
            minkowski() {
                translate([0,0, BOTTOM_ROUND/2])
                    cylinder(h = 0.01, r = BODY_R - 1);
                sphere(r = BOTTOM_ROUND/2);
            }
        }

        // slight cavity (thin wall suggestion)
        translate([0,0,BOTTOM_ROUND])
            cylinder(h = BODY_H, r = BODY_R - 2);
    }
}

// ------------ Lid + spout + handle ------------
module lid_assembly() {
    union() {
        lid_ring();
        spout();
        handle();
    }
}

// gray ring that sits on top of body
module lid_ring() {
    difference() {
        cylinder(h = LID_H, r = BODY_R + LID_OVER);
        translate([0,0,-0.5])
            cylinder(h = LID_H+1, r = BODY_R - 1);
    }
}

// small cylindrical spout at one side
module spout() {
    // place spout slightly toward camera (Y+) and to the side (X+)
    translate([BODY_R - 3, 10, LID_H])
        cylinder(h = SPOUT_H, r = SPOUT_R);
}

// flip handle approximated as a thin curved-ish bar
module handle() {
    // base of handle anchored near spout side
    // we'll just do a simple arched bar made from hull of two cubes
    translate([BODY_R + 1, 0, LID_H + SPOUT_H - 8]) {
        hull() {
            // back anchor near lid
            translate([0, -HANDLE_W/2, 0])
                cube([HANDLE_TH, HANDLE_W, HANDLE_TH], center = false);

            // front tip lifted slightly (like a lever)
            translate([HANDLE_L, -HANDLE_W/2, 10])
                cube([HANDLE_TH, HANDLE_W, HANDLE_TH], center = false);
        }
    }
}