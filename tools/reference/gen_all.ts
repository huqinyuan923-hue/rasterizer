import { run as runM1 } from "./gen_m1.js";
import { run as runM2 } from "./gen_m2.js";
import { run as runM3 } from "./gen_m3.js";
import { run as runM4 } from "./gen_m4.js";
import { run as runM5 } from "./gen_m5.js";
import { run as runM6 } from "./gen_m6.js";
import { run as runM7 } from "./gen_m7.js";
import { run as runP2Scenegraph } from "./gen_p2_scenegraph.js";
import { run as runP2Lights } from "./gen_p2_lights.js";
import { run as runP2Mip } from "./gen_p2_mip.js";
import { run as runP2Obj } from "./gen_p2_obj.js";
import { run as runP2Cull } from "./gen_p2_cull.js";
import { run as runP2Frustum } from "./gen_p2_frustum.js";
import { run as runP2Camera } from "./gen_p2_camera.js";
import { run as runP2Sort } from "./gen_p2_sort.js";

// 注意:gen_p2_anchor.ts(升级前锚点)故意不进 gen_all —— 重新生成锚点会让
// 「场景图等价」对拍退化为自我引用;锚点只允许一次性生成并冻结。

runM1();
runM2();
runM3();
runM4();
runM5();
runM6();
runM7();
runP2Scenegraph();
runP2Lights();
runP2Mip();
runP2Obj();
runP2Cull();
runP2Frustum();
runP2Camera();
runP2Sort();
console.log("all golden images regenerated");
