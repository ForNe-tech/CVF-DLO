using Newtonsoft.Json;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class LocatableCameraUtils : MonoBehaviour
{
    private static int IMG_Width = 896;
    private static int IMG_Height = 504;
    private static Color32[] pixels;
    private static Matrix4x4 cameraToWorldMatrix;
    private static Matrix4x4 projectionMatrix;

    public GameObject Scene;

    private bool once = true;

    private string totalDir;

    private Color randomColor = Color.red;

    public int layerID = 8;
    public LayerMask LayerCollider;

    // Start is called before the first frame update
    void Start()
    {
        totalDir = @"G:/Research/DLOs Detection/CVF3D-DLO-main/data/LAB_imgs_1028_DLO";
        GetScenePose(totalDir + "/ScenePose.json");
    }

    // Update is called once per frame
    void Update()
    {
        if (once)
        {
            once = false;
            Debug.Log("Start Project!");
            string[] filePaths = Directory.GetFiles(totalDir + "/route_seg");
            int lengthToRemove = 4;
            foreach (string filePath in filePaths)
            {
                string pngName = Path.GetFileName(filePath);
                int endIndex = pngName.Length - lengthToRemove > 0 ? pngName.Length - lengthToRemove : 0;
                string fileName = pngName.Substring(0, endIndex);
                if (fileName == "temp0")
                {
                    randomColor = Color.red;
                }
                if (fileName == "temp19")
                {
                    randomColor = Color.green;
                }
                if (fileName == "temp20")
                {
                    randomColor = Color.blue;
                }
                ProjectRouteToSpace(totalDir, fileName);

                layerID += 1;
            }
            //ProjectRouteToSpace(totalDir, "temp0");
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
    }

    public static Vector3 PixelCoordToWorldCoord(Matrix4x4 cameraToWorldMatrix, Matrix4x4 projectionMatrix, Vector2 pixelCoordinates)
    {
        pixelCoordinates = ConvertPixelCoordsToScaleCoords(pixelCoordinates);

        float focalLengthX = projectionMatrix.GetColumn(0).x;
        float focalLengthY = projectionMatrix.GetColumn(1).y;
        float centerX = projectionMatrix.GetColumn(2).x;
        float centerY = projectionMatrix.GetColumn(2).y;

        float normFactor = projectionMatrix.GetColumn(2).z;
        centerX = centerX / normFactor;
        centerY = centerY / normFactor;

        Vector3 dirRay = new Vector3((pixelCoordinates.x - centerX) / focalLengthX, (pixelCoordinates.y - centerY) / focalLengthY, 1.0f / normFactor);
        Vector3 direction = new Vector3(Vector3.Dot(cameraToWorldMatrix.GetRow(0), dirRay), Vector3.Dot(cameraToWorldMatrix.GetRow(1), dirRay), Vector3.Dot(cameraToWorldMatrix.GetRow(2), dirRay));

        return direction;
    }

    public static Vector2 ConvertPixelCoordsToScaleCoords(Vector2 pixelCoords)
    {
        float halfWidth = IMG_Width / 2f;
        float halfHeight = IMG_Height / 2f;

        pixelCoords.x -= halfWidth;
        pixelCoords.y -= halfHeight;

        pixelCoords = new Vector2(pixelCoords.x / halfWidth, pixelCoords.y / halfHeight * -1f);

        return pixelCoords;
    }

    public void ProjectPointToSpace(Vector3 ori, Vector3 dir)
    {
        CreateRayEntity(ori, dir);
        RaycastHit hit;
        //if (layerID > 8)
        //{
        //    string lastLayerName = LayerMask.LayerToName(layerID - 1);
        //    LayerMask LayerMaskForCollider = LayerMask.GetMask(lastLayerName);
        //    Debug.Log("层ID " + layerID + "对应层的名称：" + lastLayerName);
        //    if (Physics.Raycast(ori, dir, out hit, Mathf.Infinity, LayerMaskForCollider))
        //    {
        //        // 如果射线与平面碰撞，打印碰撞物体信息
        //        Debug.Log("Hit Point:" + hit.point.ToString("F6"));
        //        // 在场景中绘制空间点
        //        CreateSphere(hit.point);
        //    }
        //    else
        //    {
        //        Debug.Log("Don't Hit");
        //    }
        //}
        if (Physics.Raycast(ori, dir, out hit, Mathf.Infinity, LayerCollider))
        {
            // 如果射线与平面碰撞，打印碰撞物体信息
            Debug.Log("Hit Point:" + hit.point.ToString("F6"));
            // 在场景中绘制空间点
            CreateSphere(hit.point);
        }
        else
        {
            Debug.Log("Don't Hit");
        }
    }

    public void CreateRayEntity(Vector3 ori, Vector3 dir)
    {
        GameObject RayForShow = new GameObject("RayForShow");
        RayForShow.layer = layerID;

        RayForShow.AddComponent<LineRenderer>();
        LineRenderer lineRenderer = RayForShow.GetComponent<LineRenderer>();
        lineRenderer.startColor = randomColor;
        lineRenderer.endColor = randomColor;
        lineRenderer.startWidth = 0.001f;
        lineRenderer.endWidth = 0.001f;
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.positionCount = 2;
        lineRenderer.SetPosition(0, ori);
        lineRenderer.SetPosition(1, ori + dir.normalized * 1f);

        //RayForShow.transform.position = ori;
        //RayForShow.transform.rotation = Quaternion.LookRotation(dir);
        //BoxCollider collider = RayForShow.AddComponent<BoxCollider>();
        //collider.size = new Vector3(0.0001f, 0.0001f, 0.5f);
        //collider.center = new Vector3(0f, 0f, 0.5f);
    }

    public static Vector3 GetHololensCameraPosByMatrix(Matrix4x4 cameraToWorldMatrix)
    {
        Vector3 pos = new Vector3(cameraToWorldMatrix[12], cameraToWorldMatrix[13], cameraToWorldMatrix[14]);
        return pos;
    }

    public static void CreateSphere(Vector3 point)
    {
        GameObject SpacePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        SpacePoint.GetComponent<Collider>().enabled = false;
        SpacePoint.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
        SpacePoint.transform.position = point;
    }

    private static List<Vector3> RouteImageToDirections(string imgPath, string jsonPath)
    {
        string jsonContent = File.ReadAllText(jsonPath);
        Dictionary<string, string> Pose_Json = JsonToDictionary(jsonContent);
        cameraToWorldMatrix = StringToMatrix(Pose_Json["cameraToWorldMatrix"].ToString());
        projectionMatrix = StringToMatrix(Pose_Json["projectionMatrix"].ToString());

        Texture2D texture2D = new Texture2D(IMG_Width, IMG_Height);
        byte[] fileData = System.IO.File.ReadAllBytes(imgPath);
        texture2D.LoadImage(fileData);

        List<Vector3> directions = new List<Vector3>();

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
                    Vector2 pixelCoord = new Vector2(pixel_x, IMG_Height - pixel_y);
                    Debug.Log("pixelCoord:" + pixelCoord.x.ToString() + pixelCoord.y.ToString());
                    Vector3 direction = PixelCoordToWorldCoord(cameraToWorldMatrix, projectionMatrix, pixelCoord);
                    Debug.Log("direction:" + direction.x.ToString() + direction.y.ToString() + direction.z.ToString());
                    directions.Add(direction);
                }
            }
        }

        return directions;
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
        // Debug.Log("cameraToWorldMatrix:" + cameraToWorldMatix.ToString("F6"));
        return cameraToWorldMatix;
    }

    private static Dictionary<string, string> JsonToDictionary(string jsonData)
    {
        return JsonConvert.DeserializeObject<Dictionary<string, string>>(jsonData);
    }

    public void ProjectRouteToSpace(string Dir, string file)
    {
        List<Vector3> dirs = RouteImageToDirections(Dir + "/route/" + file + ".png", Dir + "/json/" + file + ".json");
        Vector3 ori = GetHololensCameraPosByMatrix(cameraToWorldMatrix);
        foreach (Vector3 dir in dirs)
        {
            ProjectPointToSpace(ori, dir);
        }
    }

}
