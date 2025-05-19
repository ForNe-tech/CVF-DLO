using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class RouteToSpace : MonoBehaviour
{
    public static int IMG_Width;
    public static int IMG_Height;

    private static Color32[] pixels;

    public GameObject Scene;

    private bool once = true;

    private void Awake()
    {
        
    }

    // Start is called before the first frame update
    void Start()
    {
        string totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_0621_PT";
        GetScenePose(totalDir + "/ScenePose.json");
    }

    // Update is called once per frame
    void Update()
    {
        if (once)
        {
            once = false;
            Debug.Log("projectionMatrix_virtual_main:" + Camera.main.projectionMatrix.ToString("F6"));
            Debug.Log("Start Project!");
            string totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_0621_PT";
            string[] filePaths = Directory.GetFiles(totalDir + "/route");
            int lengthToRemove = 4;
            //foreach (string filePath in filePaths)
            //{
            //    string pngName = Path.GetFileName(filePath);
            //    int endIndex = pngName.Length - lengthToRemove > 0 ? pngName.Length - lengthToRemove : 0;
            //    string fileName = pngName.Substring(0, endIndex);
            //    //ProjectRouteToSpace(totalDir, fileName);
            //    ProjectRouteToSpace1(totalDir, fileName);
            //    //ProjectClipPointToSpace(totalDir, fileName);
            //}
            //ProjectRouteToSpace(totalDir, "temp0");
            //ProjectRouteToSpace1(totalDir, "temp0");
            ProjectClipPointToSpace(totalDir, "temp0");
        }

    }

    public void GetScenePose(string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Scene_Pose = JsonToDictionary(jsonContent);
        Vector3 Scene_P = StringToVector3(Scene_Pose["Scene_P"].ToString());
        Quaternion Scene_Q = StringToQuaternion(Scene_Pose["Scene_Q"].ToString());
        Vector3 Scene_Scale = new Vector3(1.008004f, 1.008004f, 1.008004f);
        Scene.transform.position = Scene_P;
        Scene.transform.rotation = Scene_Q;
        // Scene.transform.localScale = Scene_Scale;
        // UpdateBoxCollider();
        
    }

    public void UpdateBoxCollider()
    {
        BoxCollider[] boxColliders = Scene.GetComponentsInChildren<BoxCollider>(true);
        foreach (BoxCollider boxCollider in boxColliders)
        {
            
        }

    }

    public void GetCameraPose(string cameraPath)
    {
        string jsonContent = File.ReadAllText(cameraPath);
        Dictionary<string, string> Camera_Pose = JsonToDictionary(jsonContent);
        Vector3 Camera_P = StringToVector3(Camera_Pose["Camera_P"].ToString());
        Quaternion Camera_Q = StringToQuaternion(Camera_Pose["Camera_Q"].ToString());
        Camera.main.transform.position = Camera_P;
        Camera.main.transform.rotation = Camera_Q;
        Matrix4x4 projectionMatrix = StringToMatrix(Camera_Pose["projectionMatrix_virtual"].ToString());
        Camera.main.projectionMatrix = projectionMatrix;
        Debug.Log("projectionMatrix:" + Camera.main.projectionMatrix.ToString("F6"));
    }

    public void GetCameraPoseFromMatrix(string cameraPath)
    {
        string jsonContent = File.ReadAllText(cameraPath);
        Dictionary<string, string> Camera_Pose = JsonToDictionary(jsonContent);
        Matrix4x4 cameraToWorldMatrix = StringToMatrix(Camera_Pose["cameraToWorldMatrix"].ToString());
        Vector3 Camera_P = cameraToWorldMatrix.GetColumn(3);
        Quaternion Camera_Q = ExtractRotationFromMatrix(cameraToWorldMatrix);
        Debug.Log("Camera_P:" + Camera_P.ToString("F6"));
        Debug.Log("Camera_Q:" + Camera_Q.ToString("F6"));
        //Vector3 Camera_P0 = StringToVector3(Camera_Pose["Camera_P"].ToString());
        //Vector3 Camera_P_Error = Camera_P - Camera_P0;
        //Debug.Log("Camera_P_Error:" + Camera_P_Error.ToString("F6"));
        Camera.main.transform.position = Camera_P;
        Camera.main.transform.rotation = Camera_Q;
        Matrix4x4 projectionMatrix = StringToMatrix(Camera_Pose["projectionMatrix_virtual"].ToString());
        Camera.main.projectionMatrix = projectionMatrix;
        Debug.Log("projectionMatrix:" + Camera.main.projectionMatrix.ToString("F6"));
    }

    private static Quaternion ExtractRotationFromMatrix(Matrix4x4 cameraToWorldMatrix)
    {
        Vector3 rotationX = cameraToWorldMatrix.GetColumn(0);
        Vector3 rotationY = cameraToWorldMatrix.GetColumn(1);
        Vector3 rotationZ = cameraToWorldMatrix.GetColumn(2);
        Quaternion rotation = Quaternion.LookRotation(-rotationZ, rotationY);
        return rotation;
    }

    private static Vector3 StringToVector3(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        return new Vector3(float.Parse(s[0]), float.Parse(s[1]), float.Parse(s[2]));
    }

    private static Quaternion StringToQuaternion(string str)
    {
        str = str.Replace("(", " ").Replace(")", " "); //将字符串中"("和")"替换为" "
        string[] s = str.Split(',');
        Quaternion q = new Quaternion();
        q.w = float.Parse(s[0]);
        q.x = float.Parse(s[1]);
        q.y = float.Parse(s[2]);
        q.z = float.Parse(s[3]);
        return q;
    }

    private static Matrix4x4 StringToMatrix(string str)
    {
        str = str.Replace("\n", ",").Replace("\t", ",");
        string[] s = str.Split(',');
        Matrix4x4 cameraToWorldMatix = new Matrix4x4();
        for (int col = 0; col < 4; col++)
        {
            for (int row = 0; row < 4; row++)
            {
                cameraToWorldMatix[row, col] = float.Parse(s[row * 4 + col]);
            }
        }
        Debug.Log("cameraToWorldMatrix:" + cameraToWorldMatix.ToString("F6"));
        return cameraToWorldMatix;
    }

    

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
    }

    public void ProjectRouteToSpace(string Dir, string file)
    {
        GetCameraPoseFromMatrix(Dir + "/json/" + file + ".json");
        List<Vector3> ClipPoints = ReadRouteImage(Dir + "/route/" + file + ".png");
        // Debug.Log("Here");
        foreach (Vector3 p in ClipPoints)
        {
            ProjectPointToSpace(p);
        }
    }

    public void ProjectRouteToSpace1(string Dir, string file)
    {
        GetCameraPose(Dir + "/json/" + file + ".json");
        List<Vector3> ClipPoints = ReadRouteImage(Dir + "/route/" + file + ".png");
        // Debug.Log("Here");
        foreach (Vector3 p in ClipPoints)
        {
            ProjectPointToSpace(p);
        }
    }

    public void ProjectPointToSpace(Vector3 clipPoint)
    {
        Debug.Log("clip Point:" + clipPoint.ToString("F6"));
        Ray ray = Camera.main.ViewportPointToRay(clipPoint);
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, Mathf.Infinity))
        {
            // 如果射线与平面碰撞，打印碰撞物体信息
            Debug.Log("Hit Point:" + hit.point.ToString("F6"));
            // 在场景中绘制空间点
            CreateSphere(hit.point);
            ShowRay(ray);
        }
        else
        {
            Debug.Log("Don't Hit");
        }
    }

    public void ProjectClipPointToSpace(string Dir, string file)
    {
        GetCameraPose(Dir + "/json/" + file + ".json");
        string jsonContent = File.ReadAllText(Dir + "/json/" + file + ".json");
        Dictionary<string, string> Dict_mask = JsonToDictionary(jsonContent);
        int mask_num = Convert.ToInt32(Dict_mask["mask_num"]);
        for (int i = 1; i < mask_num+1; i++)
        {
            string single_mask_string = Dict_mask["mask_" + i.ToString()];
            Dictionary<string, string> single_mask = JsonToDictionary(single_mask_string);
            Vector3 clipPoint = new Vector3();
            clipPoint.x = float.Parse(single_mask["mask_x"]);
            clipPoint.y = float.Parse(single_mask["mask_y"]);
            clipPoint.z = float.Parse(single_mask["mask_z"]);
            ProjectPointToSpace(clipPoint);
        }
    }

    public void CreateSphere(Vector3 point)
    {
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.GetComponent<Collider>().enabled = false;
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
    }

    public void ShowRay(Ray ray)
    {
        float rayLength = 10f;
        Color rayColor = Color.yellow;
        Debug.DrawRay(ray.origin, ray.direction * rayLength, rayColor);
    }

    private static List<Vector3> ReadRouteImage(string imgPath)
    {
        Texture2D texture2D = new Texture2D(IMG_Width, IMG_Height);
        byte[] fileData = System.IO.File.ReadAllBytes(imgPath);
        texture2D.LoadImage(fileData);

        //// 参数0614
        //float linreg_x_1 = 1.62677354f;
        //float linreg_x_2 = -0.30536182f;
        //float linreg_y_1 = 1.40746227f;
        //float linreg_y_2 = -0.19553006f;

        //// 参数0617_PT
        //float linreg_x_1 = 1.63978181f;
        //float linreg_x_2 = -0.30750806f;
        //float linreg_y_1 = 1.42669546f;
        //float linreg_y_2 = -0.23591232f;

        // 参数0621_PT
        float linreg_x_1 = 1.64243742f;
        float linreg_x_2 = -0.32366764f;
        float linreg_y_1 = 1.42407627f;
        float linreg_y_2 = -0.19618871f;

        List<Vector3> ClipPoints = new List<Vector3>();

        if (texture2D != null)
        {
            pixels = texture2D.GetPixels32();
            // Debug.Log("Pixels:" + pixels);

            for (int i = 0; i < pixels.Length; i++)
            {
                // Debug.Log("pixels[" + i.ToString() + "]:" + pixels[i]);
                if (pixels[i].r != 0)
                {
                    int pixel_x = i % texture2D.width;
                    int pixel_y = i / texture2D.width;
                    Debug.Log("pixel Point:" + pixel_x.ToString() + " " + pixel_y.ToString() + " " + texture2D.width.ToString());
                    float ratio_x = (float)pixel_x / (float)texture2D.width;
                    float ratio_y = (float)pixel_y / (float)texture2D.height;
                    float clip_x = linreg_x_1 * ratio_x + linreg_x_2;
                    float clip_y = linreg_y_1 * ratio_y + linreg_y_2;
                    ClipPoints.Add(new Vector3(clip_x, clip_y, 1.6f));
                }
            }
        }
        
        return ClipPoints;
    }
}
