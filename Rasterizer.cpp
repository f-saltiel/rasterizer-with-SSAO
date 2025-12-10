#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>
#include <sstream>
#define M_PI 3.14159265358979323846

using namespace std;

// =================== Canvas / Projection ===================
const int Cw = 400;
const int Ch = 400;
const double Vw = 1.0;
const double Vh = 1.0;
const double d  = 1.0;
const double EPS = 1e-9;

const bool ENABLE_BACKFACE_CULLING = true;
double depthBuffer[Ch][Cw];

// =================== Basic Types ===================
struct Color {
    uint8_t r,g,b;
    Color(uint8_t R=255,uint8_t G=255,uint8_t B=255):r(R),g(G),b(B){}
};

struct Vec3 {
    double x,y,z;
    Vec3(double X=0,double Y=0,double Z=0):x(X),y(Y),z(Z){}
    Vec3 operator+(const Vec3&o)const{return Vec3(x+o.x,y+o.y,z+o.z);}
    Vec3 operator-(const Vec3&o)const{return Vec3(x-o.x,y-o.y,z-o.z);}
    Vec3 operator*(double s)const{return Vec3(x*s,y*s,z*s);}
    Vec3 operator/(double s)const{return Vec3(x/s,y/s,z/s);}
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    double dot(const Vec3&o)const{return x*o.x+y*o.y+z*o.z;}
    double length()const{return sqrt(x*x+y*y+z*z);}
    Vec3 cross(const Vec3& o) const {return Vec3(y*o.z - z*o.y,z*o.x - x*o.z,x*o.y - y*o.x);}
};
static Vec3 normalize(const Vec3& v){ double L=v.length(); return (L<1e-12)?Vec3(0,0,0):v/L; }

// =================== SSAO Settings ===================
const int SSAO_KERNEL_SIZE = 32;      // number of sample directions
const int SSAO_NOISE_DIM    = 4;      // 4x4 noise "texture"
const double SSAO_RADIUS    = 0.5;    // sampling radius in view space
const double SSAO_BIAS      = 0.025;  // bias to avoid self-occlusion

Vec3 normalBuffer[Ch][Cw];
Vec3 colorBuffer[Ch][Cw];
Vec3 posBuffer[Ch][Cw];
double ssaoBuffer[Ch][Cw];
double ssaoBlurBuffer[Ch][Cw];

vector<Vec3> ssaoKernel;
vector<Vec3> ssaoNoise;

struct Vec2 {
    double x,y;
    Vec2(double X=0,double Y=0):x(X),y(Y){}
};

// =================== Ambient, Lighting, Materials ===================
const Vec3 GLOBAL_AMBIENT(0.2, 0.2, 0.2);

enum class LightType { POINT, DIRECTIONAL };

struct Light {
    LightType type;
    Vec3 color;       // RGB in [0,1].
    double intensity; // Scalar multiplier.
    Vec3 direction;   // Used if DIRECTIONAL.
    Vec3 position;    // Used if POINT.
    double att_k1, att_k2;

    static Light Directional(const Vec3& dir, const Vec3& col, double I=1.0){
        Light L; L.type=LightType::DIRECTIONAL; L.direction=normalize(dir); L.color=col; L.intensity=I;
        L.att_k1=0.0; L.att_k2=0.0; return L;
    }
    static Light Point(const Vec3& pos, const Vec3& col, double I=1.0, double k1=0.09, double k2=0.032){
        Light L; L.type=LightType::POINT; L.position=pos; L.color=col; L.intensity=I;
        L.att_k1=k1; L.att_k2=k2; return L;
    }
};

struct Material {
    double ka;        // Ambient multiplier.
    double kd;        // Diffuse multiplier.
    double ks;        // Specular multiplier.
    double shininess; // Phong exponent.
    Material(double Ka=1.0,double Kd=1.0,double Ks=0.6,double Ns=32.0)
        :ka(Ka),kd(Kd),ks(Ks),shininess(Ns){}
};

// =================== Image Buffer ===================
class Image {
public:
    int W,H; vector<Color> pix;
    Image(int w,int h,Color bg=Color(255,255,255)):W(w),H(h),pix(w*h,bg){}
    void PutPixelScreen(int sx,int sy,Color c){
        if(sx>=0&&sx<W&&sy>=0&&sy<H) pix[sy*W+sx]=c;
    }
    void PutPixelCanvas(int x,int y,Color c){
        int sx=x+W/2; int sy=(H/2-1)-y; PutPixelScreen(sx,sy,c);
    }
    void PutPixelCanvas(double x,double y,Color c){ PutPixelCanvas((int)round(x),(int)round(y),c); }
    void SavePPM(const string& path) const{
        ofstream f(path,ios::binary); f<<"P6\n"<<W<<" "<<H<<"\n255\n";
        for(const Color&p:pix){ f.put((char)p.r); f.put((char)p.g); f.put((char)p.b); }
    }
};

// =================== Drawing Helpers ===================
vector<double> Interpolate(double i0,double d0,double i1,double d1){
    vector<double> values;
    if (abs(i1 - i0) < EPS) { values.push_back(d0); return values; }
    double a=(d1-d0)/(i1-i0), d=d0;
    for(int i=(int)i0;i<=(int)i1;i++){ values.push_back(d); d+=a; }
    return values;
}

void DrawLine(Image& img, Vec2 P0, Vec2 P1, Color color){
    int x0=(int)round(P0.x), y0=(int)round(P0.y);
    int x1=(int)round(P1.x), y1=(int)round(P1.y);
    if (abs(x1-x0)>abs(y1-y0)){
        if(x0>x1){ swap(x0,x1); swap(y0,y1); }
        auto ys = Interpolate(x0,y0,x1,y1);
        for(int x=x0;x<=x1;x++) img.PutPixelScreen(x,(int)round(ys[x-x0]),color);
    }else{
        if(y0>y1){ swap(x0,x1); swap(y0,y1); }
        auto xs = Interpolate(y0,x0,y1,x1);
        for(int y=y0;y<=y1;y++) img.PutPixelScreen((int)round(xs[y-y0]),y,color);
    }
}

void DrawWireframeTriangle(Image& img, Vec2 p0, Vec2 p1, Vec2 p2, Color color){
    DrawLine(img,p0,p1,color); DrawLine(img,p1,p2,color); DrawLine(img,p2,p0,color);
}

// =================== Geometry / Camera ===================
struct Plane{ Vec3 normal; double D; Plane(Vec3 n=Vec3(),double d=0):normal(n),D(d){} };
struct Transform{ double scale; Vec3 rotation, translation;
    Transform(double s=1.0, Vec3 r=Vec3(), Vec3 t=Vec3()):scale(s),rotation(r),translation(t){} };

struct Triangle3D { Vec3 A,B,C; Color color;
    Triangle3D():A(),B(),C(),color(128,128,128) {}
    Triangle3D(const Vec3&a,const Vec3&b,const Vec3&c, Color col):A(a),B(b),C(c),color(col){}
};

// =================== Triangle2D, Positions, Z ===================
struct Triangle2D {
    Vec2 A,B,C;
    double zA,zB,zC;          // Camera-space Z per vertex.
    Vec3 posA,posB,posC;      // Camera-space positions.
    Vec3 normal;              // Face normal in camera space.
    Color color;
    Triangle2D():A(),B(),C(),zA(0),zB(0),zC(0),posA(),posB(),posC(),normal(),color(128,128,128){}
    Triangle2D(const Vec2&a,const Vec2&b,const Vec2&c, Color col,
               const Vec3& pa,const Vec3& pb,const Vec3& pc,
               double za,double zb,double zc)
        :A(a),B(b),C(c),
         zA(za),zB(zb),zC(zc),posA(pa),posB(pb),posC(pc),color(col){
        Vec3 U=pb-pa, V=pc-pa;
        Vec3 N( U.y*V.z - U.z*V.y, U.z*V.x - U.x*V.z, U.x*V.y - U.y*V.x );
        normal = normalize(N);
    }
};

enum RenderMode { WIREFRAME, FILLED };

Vec2 ViewportToCanvas(double x,double y){
    double cx = x * Cw / Vw + Cw/2.0;
    double cy = -y * Ch / Vh + Ch/2.0;
    return Vec2(cx,cy);
}
Vec2 ProjectVertex(const Vec3& v){
    double z = v.z; if (z<=EPS) z=EPS;
    return ViewportToCanvas(v.x * d / z, v.y * d / z);
}

Vec3 ScaleVec(const Vec3& v,double s){
    return Vec3(v.x*s,v.y*s,v.z*s);
}

Vec3 RotateX(const Vec3& v,double a){
    double r=a*M_PI/180.0,c=cos(r),s=sin(r);
    return Vec3(v.x, v.y*c + v.z*s, -v.y*s + v.z*c);
}

Vec3 RotateY(const Vec3& v,double a){
    double r=a*M_PI/180.0,c=cos(r),s=sin(r);
    return Vec3(v.x*c - v.z*s, v.y, v.x*s + v.z*c);
}

Vec3 RotateZ(const Vec3& v,double a){
    double r=a*M_PI/180.0,c=cos(r),s=sin(r);
    return Vec3(v.x*c + v.y*s, -v.x*s + v.y*c, v.z);
}

Vec3 Translate(const Vec3& v,const Vec3&t){
    return Vec3(v.x+t.x,v.y+t.y,v.z+t.z);
}

Vec3 ApplyTransform(const Vec3& v,const Transform& tr){
    Vec3 r = ScaleVec(v,tr.scale);
    r = RotateZ(r,tr.rotation.z);
    r = RotateY(r,tr.rotation.y);
    r = RotateX(r,tr.rotation.x);
    r = Translate(r,tr.translation);
    return r;
}

struct Camera { Vec3 translation, rotation;
    Camera(Vec3 t=Vec3(),Vec3 r=Vec3()):translation(t),rotation(r){}
};
Vec3 ApplyCamera(const Vec3& v,const Camera& cam){
    Vec3 r = v - cam.translation;
    r = RotateZ(r,-cam.rotation.z);
    r = RotateY(r,-cam.rotation.y);
    r = RotateX(r,-cam.rotation.x);
    return r;
}

// =================== Clipping ===================
double SignedDistance(const Plane& pl,const Vec3& p){ return pl.normal.dot(p)+pl.D; }
Vec3 IntersectSegmentPlane(const Vec3& A,const Vec3& B,const Plane& pl){
    Vec3 BA=B-A; double denom=pl.normal.dot(BA);
    if (fabs(denom)<EPS) return A;
    double t = -(pl.normal.dot(A)+pl.D)/denom; t=max(0.0,min(1.0,t));
    return A + BA*t;
}
vector<Triangle3D> ClipTriangleAgainstPlane(const Triangle3D& tri,const Plane& pl){
    vector<Triangle3D> out;
    double dA=SignedDistance(pl,tri.A), dB=SignedDistance(pl,tri.B), dC=SignedDistance(pl,tri.C);
    bool inA=dA>=0, inB=dB>=0, inC=dC>=0;
    if (inA&&inB&&inC){ out.push_back(tri); return out; }
    if (!inA&&!inB&&!inC){ return out; }
    if (inA && !inB && !inC){
        Vec3 Bp=IntersectSegmentPlane(tri.A,tri.B,pl), Cp=IntersectSegmentPlane(tri.A,tri.C,pl);
        out.emplace_back(tri.A,Bp,Cp,tri.color);
    } else if (inB && !inA && !inC){
        Vec3 Ap=IntersectSegmentPlane(tri.B,tri.A,pl), Cp=IntersectSegmentPlane(tri.B,tri.C,pl);
        out.emplace_back(tri.B,Ap,Cp,tri.color);
    } else if (inC && !inA && !inB){
        Vec3 Ap=IntersectSegmentPlane(tri.C,tri.A,pl), Bp=IntersectSegmentPlane(tri.C,tri.B,pl);
        out.emplace_back(tri.C,Ap,Bp,tri.color);
    } else {
        if (!inA){
            Vec3 A_p=IntersectSegmentPlane(tri.B,tri.A,pl), B_p=IntersectSegmentPlane(tri.C,tri.A,pl);
            out.emplace_back(tri.B,tri.C,A_p,tri.color);
            out.emplace_back(tri.C,B_p,A_p,tri.color);
        } else if (!inB){
            Vec3 A_p=IntersectSegmentPlane(tri.A,tri.B,pl), B_p=IntersectSegmentPlane(tri.C,tri.B,pl);
            out.emplace_back(tri.A,tri.C,A_p,tri.color);
            out.emplace_back(tri.C,B_p,A_p,tri.color);
        } else {
            Vec3 A_p=IntersectSegmentPlane(tri.A,tri.C,pl), B_p=IntersectSegmentPlane(tri.B,tri.C,pl);
            out.emplace_back(tri.A,tri.B,A_p,tri.color);
            out.emplace_back(tri.B,B_p,A_p,tri.color);
        }
    }
    return out;
}
vector<Triangle3D> ClipTrianglesAgainstPlane(const vector<Triangle3D>& tris,const Plane& pl){
    vector<Triangle3D> out;
    for (const auto& t: tris){
        auto v = ClipTriangleAgainstPlane(t,pl);
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

// =================== Bounding Sphere and Frustum ===================
pair<Vec3,double> ComputeModelBoundingSphere(const vector<Vec3>& verts){
    if (verts.empty()) return {Vec3(),0.0};
    Vec3 c(0,0,0); for(const auto& v:verts) c=c+v; c=c*(1.0/verts.size());
    double r=0; for(const auto& v:verts) r=max(r,(v-c).length());
    return {c,r};
}
vector<Plane> MakeFrustumPlanes(double Vw,double Vh,double d,double near){
    vector<Plane> P;
    P.emplace_back(Vec3(0,0,1), -near);            // Near.
    P.emplace_back(Vec3(d,0,Vw/2.0), 0.0);         // Left.
    P.emplace_back(Vec3(-d,0,Vw/2.0), 0.0);        // Right.
    P.emplace_back(Vec3(0,-d,Vh/2.0), 0.0);        // Top.
    P.emplace_back(Vec3(0, d,Vh/2.0), 0.0);        // Bottom.
    return P;
}

// =================== Model / Instance ===================
class Model{
public:
    string name;
    vector<Vec3> vertices;
    vector<tuple<int,int,int,Color>> triangles;
    pair<Vec3,double> bounding_sphere;
    Model(string n, vector<Vec3> v, vector<tuple<int,int,int,Color>> t)
        :name(n),vertices(v),triangles(t){
        bounding_sphere=ComputeModelBoundingSphere(vertices);
    }
};

class Instance{
public:
    Model* model;
    Transform transform;
    Material material;
    Instance(Model* m, Transform t, Material mat=Material())
        :model(m),transform(t),material(mat){}
};

// =================== Model Makers ===================
Model readModelFromText(const string& name,const string& vertPath,const string& facePath, Color color){
    ifstream vf(vertPath), ff(facePath);
    if(!vf.is_open()) throw runtime_error("Failed to open vertex file: "+vertPath);
    if(!ff.is_open()) throw runtime_error("Failed to open face file: "+facePath);
    vector<Vec3> verts; vector<tuple<int,int,int,Color>> tris;
    int vi; float x,y,z; char colon;
    while(vf>>vi>>colon>>x>>y>>z){ verts.emplace_back(x,y,z); }
    int fi,a,b,c;
    while(ff>>fi>>colon>>a>>b>>c){ tris.emplace_back(a,b,c,color); }
    return Model(name, verts, tris);
}
Model readModelFromObj(const string& name,const string& filePath, Color color){
    ifstream obj(filePath);
    if(!obj.is_open()){ cerr<<"readModelFromObj: failed to open "<<filePath<<"\n";
        return Model(name,{},{}); }
    vector<Vec3> verts; vector<tuple<int,int,int,Color>> tris; string line;
    while(getline(obj,line)){
        istringstream iss(line); string tag; if(!(iss>>tag)) continue;
        if(tag=="v"){ double x,y,z; if(iss>>x>>y>>z) verts.emplace_back(x,y,z); }
        else if(tag=="f"){
            vector<int> idx; string tok;
            while(iss>>tok){
                size_t s=tok.find('/'); string sidx=(s==string::npos)?tok:tok.substr(0,s);
                if(sidx.empty()) continue; try{ int v=stoi(sidx); idx.push_back(v-1);}catch(...){}
            }
            if(idx.size()<3) continue;
            for(size_t i=1;i+1<idx.size();++i){
                int A=idx[0],B=idx[i],C=idx[i+1];
                if(A<0||B<0||C<0) continue;
                if((size_t)A>=verts.size()||(size_t)B>=(verts.size())||(size_t)C>=verts.size()) continue;
                tris.emplace_back(A,B,C,color);
            }
        }
    }
    return Model(name, verts, tris);
}

Model makeSphereModel(const string& name, double radius, int latDiv, int lonDiv, Color baseColor){
    vector<Vec3> verts;
    vector<tuple<int,int,int,Color>> tris;

    for (int i = 0; i <= latDiv; ++i) {
        double theta = M_PI * i / latDiv; // 0 to π.
        for (int j = 0; j <= lonDiv; ++j) {
            double phi = 2.0 * M_PI * j / lonDiv; // 0 to 2π.
            double x = radius * sin(theta) * cos(phi);
            double y = radius * cos(theta);
            double z = radius * sin(theta) * sin(phi);
            verts.emplace_back(x, y, z);
        }
    }

    for (int i = 0; i < latDiv; ++i) {
        for (int j = 0; j < lonDiv; ++j) {
            int a = i * (lonDiv + 1) + j;
            int b = a + lonDiv + 1;
            int c = a + 1;
            int d = b + 1;

            // Two triangles per quad.
            tris.emplace_back(a, c, b, baseColor);
            tris.emplace_back(c, d, b, baseColor);
        }
    }

    return Model(name, verts, tris);
}

// =================== Shading ===================
static double clamp01(double v){ if(v<0) return 0; if(v>1) return 1; return v; }
static Vec3 ColorToVec3(const Color& c){ return Vec3(c.r/255.0, c.g/255.0, c.b/255.0); }

Vec3 ShadePhongAtPoint(const Vec3& pos_cam, const Vec3& normal_in, const Vec3& baseColor, const vector<Light>& lights, const Material& mat){
    Vec3 N = normalize(normal_in);
    if (N.length()==0) return baseColor; // Degenerate, show base.

    if (mat.kd == 0.0 && mat.ks == 0.0 && mat.ka == 0.0) {
        return baseColor; // Skip lighting, show raw color
    }

    Vec3 V = normalize(Vec3(0,0,0) - pos_cam);

    // Ambient term.
    Vec3 ambient = GLOBAL_AMBIENT * mat.ka;
    ambient = Vec3(ambient.x*baseColor.x, ambient.y*baseColor.y, ambient.z*baseColor.z);

    Vec3 diffuse(0,0,0), spec(0,0,0);

    for (const auto& Lgt : lights){
        if (Lgt.type == LightType::DIRECTIONAL){
            Vec3 L = normalize(Lgt.direction);
            double ndotl = max(0.0, N.dot(L));
            Vec3 diff = Lgt.color * (mat.kd * ndotl * Lgt.intensity);

            Vec3 H = normalize(L + V);
            double ndoth = max(0.0, N.dot(H));
            Vec3 spc = Lgt.color * (mat.ks * pow(ndoth, mat.shininess) * Lgt.intensity);

            diffuse = diffuse + Vec3(diff.x*baseColor.x, diff.y*baseColor.y, diff.z*baseColor.z);
            spec = spec + spc;
        } else {
            Vec3 L = normalize(Lgt.position - pos_cam);
            double dist = (Lgt.position - pos_cam).length();
            double atten = 1.0 / (1.0 + Lgt.att_k1*dist + Lgt.att_k2*dist*dist);

            double ndotl = max(0.0, N.dot(L));
            Vec3 diff = Lgt.color * (mat.kd * ndotl * Lgt.intensity * atten);

            Vec3 H = normalize(L + V);
            double ndoth = max(0.0, N.dot(H));
            Vec3 spc = Lgt.color * (mat.ks * pow(ndoth, mat.shininess) * Lgt.intensity * atten);

            diffuse = diffuse + Vec3(diff.x*baseColor.x, diff.y*baseColor.y, diff.z*baseColor.z);
            spec = spec + spc;
        }
    }

    Vec3 out = ambient + diffuse + spec;
    return Vec3(clamp01(out.x), clamp01(out.y), clamp01(out.z));
}

// =================== Pipeline: Clip and Project ===================
vector<Triangle2D> ClipAndProjectInstance(const Instance& instance, const Camera& camera, const vector<Plane>& planes, int& totalTris, int& culledTris){
    vector<Triangle2D> projected;

    // Model -> World -> Camera.
    vector<Vec3> vw; vw.reserve(instance.model->vertices.size());
    vector<Vec3> vc; vc.reserve(instance.model->vertices.size());
    for (const auto& v : instance.model->vertices) {
        Vec3 w = ApplyTransform(v, instance.transform);
        vw.push_back(w);
        vc.push_back(ApplyCamera(w, camera));
    }

    // Bounding-sphere cull.
    auto [mc, mr] = instance.model->bounding_sphere;
    Vec3 cw = ApplyTransform(mc, instance.transform);
    Vec3 cc = ApplyCamera(cw, camera);
    double rc = fabs(instance.transform.scale) * mr;

    // Triangles in camera space.
    vector<Triangle3D> tris; tris.reserve(instance.model->triangles.size());
    for (const auto& t : instance.model->triangles) {
        int a = get<0>(t), b = get<1>(t), c = get<2>(t);
        Color col = get<3>(t);
        tris.emplace_back(vc[a], vc[b], vc[c], col);
    }

    // Clip against planes.
    for (const auto& pl : planes) {
        double ds = SignedDistance(pl, cc);
        if (ds > rc) continue; // Fully in front for sphere.
        tris = ClipTrianglesAgainstPlane(tris, pl);
        if (tris.empty()) return projected;
    }

    // Backface culling.
    vector<Triangle3D> visible;
    visible.reserve(tris.size());
    for (const auto& t : tris) {
        totalTris++;

        Vec3 U = t.B - t.A;
        Vec3 V = t.C - t.A;
        Vec3 N(U.y * V.z - U.z * V.y,
            U.z * V.x - U.x * V.z,
            U.x * V.y - U.y * V.x);
        Vec3 Nn = normalize(N);

        Vec3 triCenter = (t.A + t.B + t.C) / 3.0;
        Vec3 viewDir = normalize(triCenter * -1.0);
        double ndotv = Nn.dot(viewDir);

        if (ENABLE_BACKFACE_CULLING) {
            if (ndotv > 0.0) visible.push_back(t);
            else culledTris++;
        } else {
            visible.push_back(t); // Skip culling entirely.
        }
    }

    // Projection.
    for (const auto& t : visible) {
        Vec2 A2 = ProjectVertex(t.A), B2 = ProjectVertex(t.B), C2 = ProjectVertex(t.C);

        projected.emplace_back(
            A2, B2, C2,
            t.color,
            t.A, t.B, t.C,
            t.A.z, t.B.z, t.C.z
        );
    }

    return projected;
}

// =================== Filled Triangle ===================
void DrawFilledTriangle(Image& img, const Triangle2D& tri, const vector<Light>& lights, const Material& mat){
    Vec2 P0=tri.A, P1=tri.B, P2=tri.C;
    double z0=tri.zA, z1=tri.zB, z2=tri.zC;
    Vec3 p0=tri.posA, p1=tri.posB, p2=tri.posC;

    int x0=(int)round(P0.x), y0=(int)round(P0.y);
    int x1=(int)round(P1.x), y1=(int)round(P1.y);
    int x2=(int)round(P2.x), y2=(int)round(P2.y);

    // Sort by Y, carry all attributes.
    Vec2 q0=P0,q1=P1,q2=P2; Vec3 qp0=p0,qp1=p1,qp2=p2; double zz0=z0,zz1=z1,zz2=z2;
    int cx0=x0,cy0=y0,cx1=x1,cy1=y1,cx2=x2,cy2=y2;

    auto swapAll = [&](){
        swap(cy1,cy0); swap(cx1,cx0); swap(q1,q0); swap(qp1,qp0); swap(zz1,zz0);
    };
    if (cy1<cy0){ swapAll(); }
    if (cy2<cy0){ swap(cy2,cy0); swap(cx2,cx0); swap(q2,q0); swap(qp2,qp0); swap(zz2,zz0); }
    if (cy2<cy1){ swap(cy2,cy1); swap(cx2,cx1); swap(q2,q1); swap(qp2,qp1); swap(zz2,zz1); }
    if (cy0==cy2) return;

    // Edge X.
    auto x01=Interpolate(cy0,cx0,cy1,cx1);
    auto x12=Interpolate(cy1,cx1,cy2,cx2);
    auto x02=Interpolate(cy0,cx0,cy2,cx2);

    // 1/Z along edges.
    double iz0=1.0/zz0, iz1=1.0/zz1, iz2=1.0/zz2;
    auto iz01=Interpolate(cy0,iz0,cy1,iz1);
    auto iz12=Interpolate(cy1,iz1,cy2,iz2);
    auto iz02=Interpolate(cy0,iz0,cy2,iz2);

    // Position/Z along edges.
    Vec3 p0iz=qp0*iz0, p1iz=qp1*iz1, p2iz=qp2*iz2;
    auto px01=Interpolate(cy0,p0iz.x,cy1,p1iz.x);
    auto px12=Interpolate(cy1,p1iz.x,cy2,p2iz.x);
    auto px02=Interpolate(cy0,p0iz.x,cy2,p2iz.x);

    auto py01=Interpolate(cy0,p0iz.y,cy1,p1iz.y);
    auto py12=Interpolate(cy1,p1iz.y,cy2,p2iz.y);
    auto py02=Interpolate(cy0,p0iz.y,cy2,p2iz.y);

    auto pz01=Interpolate(cy0,p0iz.z,cy1,p1iz.z);
    auto pz12=Interpolate(cy1,p1iz.z,cy2,p2iz.z);
    auto pz02=Interpolate(cy0,p0iz.z,cy2,p2iz.z);

    // Normal/Z, flat per face.
    Vec3 n0=tri.normal, n1=tri.normal, n2=tri.normal;
    Vec3 n0iz=n0*iz0, n1iz=n1*iz1, n2iz=n2*iz2;
    auto nx01=Interpolate(cy0,n0iz.x,cy1,n1iz.x);
    auto nx12=Interpolate(cy1,n1iz.x,cy2,n2iz.x);
    auto nx02=Interpolate(cy0,n0iz.x,cy2,n2iz.x);
    auto ny01=Interpolate(cy0,n0iz.y,cy1,n1iz.y);
    auto ny12=Interpolate(cy1,n1iz.y,cy2,n2iz.y);
    auto ny02=Interpolate(cy0,n0iz.y,cy2,n2iz.y);
    auto nz01=Interpolate(cy0,n0iz.z,cy1,n1iz.z);
    auto nz12=Interpolate(cy1,n1iz.z,cy2,n2iz.z);
    auto nz02=Interpolate(cy0,n0iz.z,cy2,n2iz.z);

    // Stitch short sides.
    auto popb = [](vector<double>& v){ if(!v.empty()) v.pop_back(); };
    popb(x01); popb(iz01); popb(px01); popb(py01); popb(pz01); popb(nx01); popb(ny01); popb(nz01);

    vector<double> x012; x012.insert(x012.end(),x01.begin(),x01.end()); x012.insert(x012.end(),x12.begin(),x12.end());
    vector<double> iz012; iz012.insert(iz012.end(),iz01.begin(),iz01.end()); iz012.insert(iz012.end(),iz12.begin(),iz12.end());
    vector<double> px012; px012.insert(px012.end(),px01.begin(),px01.end()); px012.insert(px012.end(),px12.begin(),px12.end());
    vector<double> py012; py012.insert(py012.end(),py01.begin(),py01.end()); py012.insert(py012.end(),py12.begin(),py12.end());
    vector<double> pz012; pz012.insert(pz012.end(),pz01.begin(),pz01.end()); pz012.insert(pz012.end(),pz12.begin(),pz12.end());
    vector<double> nx012; nx012.insert(nx012.end(),nx01.begin(),nx01.end()); nx012.insert(nx012.end(),nx12.begin(),nx12.end());
    vector<double> ny012; ny012.insert(ny012.end(),ny01.begin(),ny01.end()); ny012.insert(ny012.end(),ny12.begin(),ny12.end());
    vector<double> nz012; nz012.insert(nz012.end(),nz01.begin(),nz01.end()); nz012.insert(nz012.end(),nz12.begin(),nz12.end());

    vector<double> x_left,x_right, iz_left,iz_right, px_left,px_right, py_left,py_right, pz_left,pz_right,
                   nx_left,nx_right, ny_left,ny_right, nz_left,nz_right;

    if (!x012.empty() && !x02.empty()){
        int m = (int)x012.size()/2;
        if (m < (int)x02.size() && x02[m] < x012[m]){
            x_left=x02; iz_left=iz02; px_left=px02; py_left=py02; pz_left=pz02; nx_left=nx02; ny_left=ny02; nz_left=nz02;
            x_right=x012; iz_right=iz012; px_right=px012; py_right=py012; pz_right=pz012; nx_right=nx012; ny_right=ny012; nz_right=nz012;
        }else{
            x_left=x012; iz_left=iz012; px_left=px012; py_left=py012; pz_left=pz012; nx_left=nx012; ny_left=ny012; nz_left=nz012;
            x_right=x02; iz_right=iz02; px_right=px02; py_right=py02; pz_right=pz02; nx_right=nx02; ny_right=ny02; nz_right=nz02;
        }

        for(int y=cy0;y<=cy2;y++){
            int idx=y-cy0;
            if(idx<0 || idx>=(int)x_left.size() || idx>=(int)x_right.size()) continue;

            int xl=(int)round(x_left[idx]), xr=(int)round(x_right[idx]);
            double il=iz_left[idx], ir=iz_right[idx];

            double pxl = (idx<(int)px_left.size())?px_left[idx]:0.0;
            double pxr = (idx<(int)px_right.size())?px_right[idx]:0.0;
            double pyl = (idx<(int)py_left.size())?py_left[idx]:0.0;
            double pyr = (idx<(int)py_right.size())?py_right[idx]:0.0;
            double pzl = (idx<(int)pz_left.size())?pz_left[idx]:0.0;
            double pzr = (idx<(int)pz_right.size())?pz_right[idx]:0.0;

            double nxl = (idx<(int)nx_left.size())?nx_left[idx]:0.0;
            double nxr = (idx<(int)nx_right.size())?nx_right[idx]:0.0;
            double nyl = (idx<(int)ny_left.size())?ny_left[idx]:0.0;
            double nyr = (idx<(int)ny_right.size())?ny_right[idx]:0.0;
            double nzl = (idx<(int)nz_left.size())?nz_left[idx]:0.0;
            double nzr = (idx<(int)nz_right.size())?nz_right[idx]:0.0;

            if (xl>xr){ swap(xl,xr); swap(il,ir); swap(pxl,pxr); swap(pyl,pyr); swap(pzl,pzr);
                        swap(nxl,nxr); swap(nyl,nyr); swap(nzl,nzr); }

            auto span_iz  = Interpolate(xl, il, xr, ir);
            auto span_px  = Interpolate(xl, pxl, xr, pxr);
            auto span_py  = Interpolate(xl, pyl, xr, pyr);
            auto span_pz  = Interpolate(xl, pzl, xr, pzr);
            auto span_nx  = Interpolate(xl, nxl, xr, nxr);
            auto span_ny  = Interpolate(xl, nyl, xr, nyr);
            auto span_nz  = Interpolate(xl, nzl, xr, nzr);

            for(int x=xl;x<=xr;x++){
                int xi=x-xl; if(xi<0||xi>=(int)span_iz.size()) continue;
                double invZ=span_iz[xi]; if(invZ<=0) continue;
                double z = 1.0/invZ;

                int sx=x, sy=y;
                if (sx<0||sx>=Cw||sy<0||sy>=Ch) continue;
                if (z >= depthBuffer[sy][sx]) continue;

                // Perspective-correct reconstruction.
                Vec3 pos_cam( span_px[xi]/invZ, span_py[xi]/invZ, span_pz[xi]/invZ );
                Vec3 n_interp( span_nx[xi]/invZ, span_ny[xi]/invZ, span_nz[xi]/invZ );

                Vec3 base = ColorToVec3(tri.color);
                Vec3 lit = ShadePhongAtPoint(pos_cam, n_interp, base, lights, mat);
                uint8_t R = (uint8_t)round(clamp01(lit.x)*255.0);
                uint8_t G = (uint8_t)round(clamp01(lit.y)*255.0);
                uint8_t B = (uint8_t)round(clamp01(lit.z)*255.0);

                depthBuffer[sy][sx] = z;
                normalBuffer[sy][sx] = normalize(n_interp);
                colorBuffer[sy][sx] = Vec3(R/255.0, G/255.0, B/255.0);
                posBuffer[sy][sx] = pos_cam;
                img.PutPixelScreen(sx,sy, Color(R,G,B));
            }
        }
    }
}

void DrawTriangle(Image& img, const Triangle2D& tri, RenderMode mode, const vector<Light>& lights, const Material& mat){
    if (mode==WIREFRAME){
        DrawWireframeTriangle(img, tri.A, tri.B, tri.C, tri.color);
    }else{
        DrawFilledTriangle(img, tri, lights, mat);
    }
}

// =================== Scene Render (Global Culling Stats) ===================
void RenderSceneWithClipping(Image& img,
                             const vector<Instance>& instances,
                             const Camera& camera,
                             const vector<Plane>& planes,
                             RenderMode mode,
                             const vector<Light>& lights)
{
    int totalTris = 0;
    int culledTris = 0;

    for (const auto& inst : instances) {
        auto projected = ClipAndProjectInstance(inst, camera, planes, totalTris, culledTris);
        for (const auto& tri : projected) {
            DrawTriangle(img, tri, mode, lights, inst.material);
        }
    }

    double percent = (100.0 * culledTris / max(1, totalTris));
    cout << "Global backface culling: " << culledTris << " / " << totalTris
         << " triangles culled (" << percent << "%)\n";
}

// =================== Debug Output ===================

void SaveDepthMap(const string& filename){
    Image img(Cw, Ch, Color(0,0,0));
    double maxDepth = -1e9, minDepth = 1e9;

    // Find min and max valid depth
    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            double d = depthBuffer[y][x];
            if (d < 1e8){
                maxDepth = max(maxDepth, d);
                minDepth = min(minDepth, d);
            }
        }
    }

    double range = maxDepth - minDepth;
    if (range < 1e-9) range = 1.0;

    // Write grayscale
    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            double d = depthBuffer[y][x];
            double norm = (d - minDepth) / range;
            norm = max(0.0, min(1.0, norm));
            uint8_t g = (uint8_t)(norm * 255.0);
            img.PutPixelScreen(x, y, Color(g, g, g));
        }
    }

    img.SavePPM(filename);
}

void SaveNormalMap(const string& filename){
    Image img(Cw, Ch, Color(0,0,0));

    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            Vec3 n = normalBuffer[y][x];
            Vec3 mapped = (n + Vec3(1,1,1)) * 0.5; // map from [-1,1] to [0,1]
            
            uint8_t R = (uint8_t)(mapped.x * 255.0);
            uint8_t G = (uint8_t)(mapped.y * 255.0);
            uint8_t B = (uint8_t)(mapped.z * 255.0);

            img.PutPixelScreen(x, y, Color(R, G, B));
        }
    }

    img.SavePPM(filename);
}

void SavePositionMap(const string& filename){
    Image img(Cw, Ch, Color(0,0,0));

    // auto-find min/max for each channel
    double minX=1e9, minY=1e9, minZ=1e9;
    double maxX=-1e9, maxY=-1e9, maxZ=-1e9;

    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            Vec3 p = posBuffer[y][x];
            minX=min(minX,p.x); maxX=max(maxX,p.x);
            minY=min(minY,p.y); maxY=max(maxY,p.y);
            minZ=min(minZ,p.z); maxZ=max(maxZ,p.z);
        }
    }

    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            Vec3 p = posBuffer[y][x];

            double nx = (p.x - minX) / (maxX - minX + 1e-9);
            double ny = (p.y - minY) / (maxY - minY + 1e-9);
            double nz = (p.z - minZ) / (maxZ - minZ + 1e-9);

            uint8_t R = (uint8_t)(nx * 255.0);
            uint8_t G = (uint8_t)(ny * 255.0);
            uint8_t B = (uint8_t)(nz * 255.0);

            img.PutPixelScreen(x, y, Color(R, G, B));
        }
    }

    img.SavePPM(filename);
}

void SaveSSAOMaskBlurred(const string& filename){
    Image img(Cw, Ch, Color(0,0,0));
    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            double v = ssaoBlurBuffer[y][x];
            v = max(0.0, min(1.0, v));
            uint8_t g = (uint8_t)(v * 255.0);
            img.PutPixelScreen(x, y, Color(g,g,g));
        }
    }
    img.SavePPM(filename);
}

// =================== SSAO Kernel + Noise Initialization ===================

double rand01(){ return rand() / (double)RAND_MAX; }

void InitSSAO(){
    ssaoKernel.clear();
    ssaoKernel.reserve(SSAO_KERNEL_SIZE);

    for(int i=0;i<SSAO_KERNEL_SIZE;i++){
        // random sample in hemisphere (view space, z >= 0)
        Vec3 sample(rand01()*2.0-1.0, rand01()*2.0-1.0, rand01());
        sample = normalize(sample);

        double scale = (double)i / (double)SSAO_KERNEL_SIZE;
        // bias samples towards the origin (more dense near the center)
        scale = 0.1 + 0.9*scale*scale;

        ssaoKernel.push_back(sample * scale);
    }

    ssaoNoise.clear();
    ssaoNoise.reserve(SSAO_NOISE_DIM * SSAO_NOISE_DIM);

    for(int i=0;i<SSAO_NOISE_DIM * SSAO_NOISE_DIM;i++){
        // random rotation in the tangent plane (z = 0)
        Vec3 noise(rand01()*2.0-1.0, rand01()*2.0-1.0, 0.0);
        ssaoNoise.push_back(noise);
    }
}

// =================== SSAO Implementation ===================

void ComputeSSAO(){
    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            Vec3 N = normalBuffer[y][x];
            Vec3 P = posBuffer[y][x];

            // skip background pixels
            if (depthBuffer[y][x] > 1e8){
                ssaoBuffer[y][x] = 1.0;
                continue;
            }

            double occlusion = 0.0;

            // choose a noise vector (4x4 tiled)
            int nx = x % SSAO_NOISE_DIM;
            int ny = y % SSAO_NOISE_DIM;
            Vec3 noise = ssaoNoise[ny * SSAO_NOISE_DIM + nx];

            // build tangent basis (TBN)
            Vec3 tangent = normalize(noise - N * N.dot(noise));
            Vec3 bitangent = normalize(N.cross(tangent));
            // TBN matrix columns
            Vec3 T = tangent;
            Vec3 B = bitangent;

            for(int i=0; i<SSAO_KERNEL_SIZE; i++){
                Vec3 sample = ssaoKernel[i];

                // rotate sample into tangent space
                Vec3 sampleVec(
                    T.x * sample.x + B.x * sample.y + N.x * sample.z,
                    T.y * sample.x + B.y * sample.y + N.y * sample.z,
                    T.z * sample.x + B.z * sample.y + N.z * sample.z
                );

                Vec3 samplePos = P + sampleVec * SSAO_RADIUS;

                // project sample to screen space
                double sx = (samplePos.x / -samplePos.z) * (Cw/2.0) + (Cw/2.0);
                double sy = (samplePos.y / -samplePos.z) * (Ch/2.0) + (Ch/2.0);

                int ix = (int)sx;
                int iy = (int)sy;

                // skip if outside screen
                if(ix<0 || ix>=Cw || iy<0 || iy>=Ch) continue;

                double sampleDepth = depthBuffer[iy][ix];
                double rangeCheck = (P - samplePos).length();

                if(sampleDepth < samplePos.z + SSAO_BIAS && rangeCheck < SSAO_RADIUS){
                    occlusion += 1.0;
                }
            }

            occlusion = 1.0 - (occlusion / SSAO_KERNEL_SIZE);
            ssaoBuffer[y][x] = occlusion;
        }
    }
}

void ComputeSSAO_Blur(){
    const int radius = 2;       // small blur radius
    const double sigma_spatial = 2.0;    // spatial Gaussian
    const double sigma_depth = 0.2;      // depth sensitivity

    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            double sumWeights = 0.0;
            double sumAO = 0.0;

            double centerDepth = depthBuffer[y][x];
            double centerAO = ssaoBuffer[y][x];

            // Skip empty background pixels
            if(centerDepth > 1e8){
                ssaoBlurBuffer[y][x] = 1.0;
                continue;
            }

            for(int dy=-radius; dy<=radius; dy++){
                for(int dx=-radius; dx<=radius; dx++){
                    int ix = x + dx;
                    int iy = y + dy;

                    if(ix<0 || ix>=Cw || iy<0 || iy>=Ch) continue;

                    double neighborDepth = depthBuffer[iy][ix];
                    double neighborAO = ssaoBuffer[iy][ix];

                    // spatial weight
                    double spatialDist2 = dx*dx + dy*dy;
                    double w_spatial = exp(-(spatialDist2) / (2*sigma_spatial*sigma_spatial));

                    // depth weight (edge-preserving)
                    double depthDiff = fabs(neighborDepth - centerDepth);
                    double w_depth = exp(-(depthDiff*depthDiff) / (2*sigma_depth*sigma_depth));

                    double weight = w_spatial * w_depth;

                    sumWeights += weight;
                    sumAO += neighborAO * weight;
                }
            }

            ssaoBlurBuffer[y][x] = sumAO / max(sumWeights, 1e-8);
        }
    }
}

void SaveSSAOMask(const string& filename){
    Image img(Cw, Ch, Color(0,0,0));

    for(int y=0; y<Ch; y++){
        for(int x=0; x<Cw; x++){
            double v = ssaoBuffer[y][x];
            v = max(0.0, min(1.0, v));
            uint8_t g = (uint8_t)(v * 255.0);
            img.PutPixelScreen(x, y, Color(g, g, g));
        }
    }

    img.SavePPM(filename);
}

// =================== Main ===================
int main(){
    cout<<"Starting 3D Rasterization Engine...\n";
    Image img(Cw,Ch, Color(255,255,255));

    srand(1337);      // fixed seed for reproducible SSAO
    InitSSAO();       // build SSAO kernel + noise

    // Sample cube model.
    vector<Vec3> cube_vertices = {
        Vec3( 1,  1,  1), Vec3(-1,  1,  1),
        Vec3(-1, -1,  1), Vec3( 1, -1,  1),
        Vec3( 1,  1, -1), Vec3(-1,  1, -1),
        Vec3(-1, -1, -1), Vec3( 1, -1, -1)
    };
    vector<tuple<int,int,int,Color>> cube_tris = {
        {0,1,2, Color(255,0,0)},   {0,2,3, Color(255,0,0)},
        {4,0,3, Color(0,255,0)},   {4,3,7, Color(0,255,0)},
        {5,4,7, Color(0,0,255)},   {5,7,6, Color(0,0,255)},
        {1,5,6, Color(255,255,0)}, {1,6,2, Color(255,255,0)},
        {4,5,1, Color(128,0,128)}, {4,1,0, Color(128,0,128)},
        {2,6,7, Color(0,255,255)}, {2,7,3, Color(0,255,255)}
    };
    Model cube_model("cube", cube_vertices, cube_tris);

    // Sphere model.
    Model sphere_model = makeSphereModel("sphere", 1.0, 48, 96, Color(200, 200, 255));

    // Skyscraper model.
    Model skyscraper_model = readModelFromObj("skyscraper", "skyscraper.obj", Color(200, 200, 255));

    // Instances, per-instance materials. (ambient, diffuse, specular, shininess[1-128])
    vector<Instance> instances;
    Material shiny(0.2, 1.0, 0.8, 64.0);
    Material plastic(0.6, 0.9, 0.2, 32.0);
    Material matte(0.9, 0.7, 0.05, 4.0);

    //instances.emplace_back(&cube_model, Transform(0.3, Vec3(0,0,0), Vec3(-1, 0, 3)), matte);
    //instances.emplace_back(&cube_model, Transform(0.5, Vec3(0,30,0), Vec3( 0.5, 1, 5)), plastic);
    //instances.emplace_back(&sphere_model, Transform(0.7, Vec3(15,0,0), Vec3(0, -1, 4)), shiny);
    instances.emplace_back(&skyscraper_model, Transform(0.05, Vec3(20,15,0), Vec3(0, -0.75, 4)), matte);

    // Camera.
    Camera camera(Vec3(0,0,0), Vec3(0,0,0));

    // Frustum planes.
    double nearz = 0.1;
    auto planes = MakeFrustumPlanes(Vw, Vh, d, nearz);

    // Lights.
    vector<Light> lights;
    lights.push_back(Light::Directional(Vec3(-0.5,-1.0,-0.3), Vec3(1.0,0.95,0.9), 1.0));
    lights.push_back(Light::Point(Vec3(0.0,2.0,2.0), Vec3(1.0,1.0,1.0), 1.0, 0.09, 0.032));

    // Clear depth-buffer.
    for(int y=0;y<Ch;y++){
        for(int x=0;x<Cw;x++){
            depthBuffer[y][x] = 1e9;
            normalBuffer[y][x] = Vec3(0,0,0);
            colorBuffer[y][x] = Vec3(1,1,1); // white default
            posBuffer[y][x] = Vec3(0,0,0);
            ssaoBuffer[y][x] = 0.0;
        }
    }


    // Render.
    RenderMode mode = FILLED; // FILLED or WIREFRAME.
    RenderSceneWithClipping(img, instances, camera, planes, mode, lights);

    // Debug.
    SaveDepthMap("debug_depth.ppm");
    SaveNormalMap("debug_normals.ppm");
    SavePositionMap("debug_positions.ppm");

    ComputeSSAO();
    SaveSSAOMask("debug_ssao_raw.ppm");

    ComputeSSAO_Blur();
    SaveSSAOMaskBlurred("debug_ssao_blur.ppm");

    // Save.
    img.SavePPM("rasterOutput.ppm");
    cout<<"Rendering complete! Output saved as 'rasterOutput.ppm'\n";
    return 0;
}
