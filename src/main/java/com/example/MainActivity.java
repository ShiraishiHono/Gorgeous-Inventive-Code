package com.example;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.text.Editable;
import android.text.InputType;
import android.text.TextWatcher;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowInsets;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * 无 Key、无网络权限的可运行原型，兼容 minSdk 24。
 * 所有地图位置、设施和路线均为演示数据，不代表真实地理位置。
 * 登录仅切换本机页面：无服务器认证，不保存账号或密码。
 * 地图接入区域由 mapModuleContainer 承载；当前挂载 DemoMapView。
 * 接入高德时还需实现 SDK 生命周期、隐私授权、权限、POI 搜索等。
 */
public class MainActivity extends Activity {
    private static final int BLUE = Color.rgb(37, 99, 235);
    private final List<Place> places = new ArrayList<>();
    private final Set<String> selected = new LinkedHashSet<>();
    private SharedPreferences prefs;
    private TextToSpeech speech;
    private boolean speechReady;
    private boolean largeText, highContrast, avoidSteps;
    private boolean entered;
    private String page = "login";
    private String query = "";
    private String focusedId = "center";
    private String feedbackDraft = "";
    private String locationDraft = "";
    private int customCounter;
    private int routeStep;
    private int ink, muted, surface, background, accent;
    private LinearLayout root, results;
    private FrameLayout mapModuleContainer;
    private DemoMapView demoMap;
    private TextView selectionStatus, focusStatus, resultsStatus;
    private EditText searchInput;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        if (getActionBar() != null) getActionBar().hide();
        prefs = getSharedPreferences("access_navigation", MODE_PRIVATE);
        largeText = prefs.getBoolean("large_text", true);
        // 新首页默认蓝白色，旧版本高对比设置不会强制覆盖新界面。
        highContrast = prefs.getBoolean("map_high_contrast", false);
        avoidSteps = prefs.getBoolean("avoid_steps", true);
        seedPlaces();
        if (state != null) {
            entered = state.getBoolean("entered", false);
            page = state.getString("page", "login");
            query = state.getString("query", "");
            focusedId = state.getString("focus", "center");
            feedbackDraft = state.getString("feedback", "");
            locationDraft = state.getString("location", "");
            routeStep = state.getInt("step", 0);
            customCounter = state.getInt("counter", 0);
            restoreCustomPlaces(state.getString("custom", "[]"));
            ArrayList<String> ids = state.getStringArrayList("selected");
            if (ids != null) for (String id : ids) if (find(id) != null) selected.add(id);
        }
        speech = new TextToSpeech(getApplicationContext(), status -> {
            if (status == TextToSpeech.SUCCESS && speech != null) {
                int result = speech.setLanguage(Locale.SIMPLIFIED_CHINESE);
                speechReady = result != TextToSpeech.LANG_MISSING_DATA
                        && result != TextToSpeech.LANG_NOT_SUPPORTED;
            }
        });
        if (!entered) showLogin();
        else if ("settings".equals(page)) showSettings();
        else if ("feedback".equals(page)) showFeedback();
        else if ("route".equals(page)) showRoute();
        else showHome();
    }

    private void seedPlaces() {
        places.add(new Place("center", "示例社区服务中心", "模拟设施：坡道入口、无障碍卫生间", .32f, .36f, false));
        places.add(new Place("park", "示例公园", "模拟设施：平缓通道、休息座椅", .67f, .26f, false));
        places.add(new Place("hospital", "示例医院", "模拟设施：无障碍电梯、服务台", .72f, .69f, false));
        places.add(new Place("station", "示例地铁站", "模拟设施：直梯入口；运行状态未核验", .28f, .72f, false));
        places.add(new Place("bridge", "示例天桥观景点", "模拟路段含台阶，不适合需避开台阶的用户", .51f, .54f, false));
    }

    private int dp(float n) { return Math.round(n * getResources().getDisplayMetrics().density); }
    private int textSize() { return largeText ? 19 : 16; }
    private void colors() {
        ink = highContrast ? Color.WHITE : Color.rgb(23, 37, 60);
        muted = highContrast ? Color.LTGRAY : Color.rgb(80, 96, 116);
        background = highContrast ? Color.BLACK : Color.rgb(241, 245, 250);
        surface = highContrast ? Color.rgb(22, 22, 22) : Color.WHITE;
        accent = highContrast ? Color.rgb(255, 226, 85) : BLUE;
    }

    private GradientDrawable rounded(int color, int radius) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color);
        shape.setCornerRadius(dp(radius));
        return shape;
    }

    private LinearLayout column() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        return box;
    }

    private LinearLayout row() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.HORIZONTAL);
        box.setGravity(Gravity.CENTER_VERTICAL);
        return box;
    }

    private void begin(String name) {
        page = name;
        colors();
        demoMap = null;
        mapModuleContainer = null;
        LinearLayout shell = column();
        shell.setBackgroundColor(background);
        // 适配 targetSdk 37 的边到边显示以及软键盘，避免内容被状态栏遮挡。
        shell.setOnApplyWindowInsetsListener((v, insets) -> {
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(
                        WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                android.graphics.Insets ime = insets.getInsets(WindowInsets.Type.ime());
                v.setPadding(bars.left, bars.top, bars.right, Math.max(bars.bottom, ime.bottom));
            } else {
                v.setPadding(insets.getSystemWindowInsetLeft(), insets.getSystemWindowInsetTop(),
                        insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            }
            return insets;
        });
        getWindow().getDecorView().setSystemUiVisibility(highContrast ? 0 : View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        root = column();
        root.setPadding(dp(16), dp(16), dp(16), dp(24));
        scroll.addView(root, new ScrollView.LayoutParams(-1, -2));
        shell.addView(scroll, new LinearLayout.LayoutParams(-1, -1));
        setContentView(shell);
        shell.requestApplyInsets();
    }

    private TextView text(LinearLayout parent, String value, int size, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(value);
        t.setTextSize(size);
        t.setTextColor(color);
        t.setLineSpacing(dp(3), 1);
        t.setPadding(0, dp(5), 0, dp(7));
        if (bold) t.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        parent.addView(t, new LinearLayout.LayoutParams(-1, -2));
        return t;
    }

    private void heading(String title, String subtitle) {
        TextView titleView = text(root, title, largeText ? 28 : 25, ink, true);
        if (Build.VERSION.SDK_INT >= 28) titleView.setAccessibilityHeading(true);
        text(root, subtitle, textSize() - 2, muted, false);
    }

    private LinearLayout card(LinearLayout parent) {
        LinearLayout box = column();
        box.setPadding(dp(14), dp(12), dp(14), dp(12));
        box.setBackground(rounded(surface, 18));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.topMargin = dp(12);
        parent.addView(box, lp);
        return box;
    }

    private Button button(String label, boolean primary, View.OnClickListener action) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(textSize());
        b.setMinHeight(dp(52));
        b.setMinimumWidth(0);
        b.setPadding(dp(12), dp(10), dp(12), dp(10));
        b.setTextColor(primary ? (highContrast ? Color.BLACK : Color.WHITE) : ink);
        b.setBackgroundTintList(null);
        b.setBackground(rounded(primary ? accent : (highContrast ? Color.DKGRAY : Color.rgb(230, 238, 250)), 12));
        b.setOnClickListener(action);
        return b;
    }

    private Button addButton(LinearLayout parent, String label, boolean primary, View.OnClickListener action) {
        Button b = button(label, primary, action);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
        lp.topMargin = dp(10);
        parent.addView(b, lp);
        return b;
    }

    private EditText input(LinearLayout parent, String label, String value, int type) {
        TextView labelView = text(parent, label, textSize(), ink, true);
        EditText e = new EditText(this);
        e.setId(View.generateViewId());
        labelView.setLabelFor(e.getId());
        e.setTextSize(textSize());
        e.setTextColor(ink);
        e.setHintTextColor(muted);
        e.setMinHeight(dp(52));
        e.setPadding(dp(12), dp(10), dp(12), dp(10));
        e.setBackground(rounded(highContrast ? Color.DKGRAY : Color.rgb(241, 245, 250), 10));
        e.setInputType(type);
        boolean multiline = (type & InputType.TYPE_TEXT_FLAG_MULTI_LINE) != 0;
        e.setSingleLine(!multiline);
        if (multiline) { e.setMinLines(3); e.setGravity(Gravity.TOP); }
        e.setText(value);
        parent.addView(e, new LinearLayout.LayoutParams(-1, -2));
        return e;
    }

    private interface Changed { void accept(String value); }
    private void watch(EditText e, Changed changed) {
        e.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) { changed.accept(s.toString()); }
            @Override public void afterTextChanged(Editable s) {}
        });
    }

    private void hideKeyboard() {
        View v = getCurrentFocus();
        if (v != null) {
            InputMethodManager manager = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
            if (manager != null) manager.hideSoftInputFromWindow(v.getWindowToken(), 0);
            v.clearFocus();
        }
    }

    private void showLogin() {
        begin("login");
        heading("无障碍导航", "让出行更从容 · 地图交互原型");
        LinearLayout intro = card(root);
        text(intro, "欢迎使用", 26, ink, true);
        text(intro, "搜索目的地、勾选地点、记录道路障碍。", textSize(), muted, false);
        LinearLayout form = card(root);
        text(form, "演示登录", 23, ink, true);
        text(form, "未接入账号服务器。任意非空演示账号和密码即可进入，不代表身份验证。请勿输入真实密码。", textSize() - 2, muted, false);
        EditText account = input(form, "演示账号", "", InputType.TYPE_CLASS_TEXT);
        account.setHint("例如 demo");
        EditText password = input(form, "演示密码", "", InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        password.setHint("请使用虚构密码");
        account.setSaveEnabled(false);
        password.setSaveEnabled(false);
        if (Build.VERSION.SDK_INT >= 26) {
            account.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO);
            password.setImportantForAutofill(View.IMPORTANT_FOR_AUTOFILL_NO);
        }
        addButton(form, "演示登录", true, v -> {
            if (account.getText().toString().trim().isEmpty()) { account.setError("请输入演示账号"); return; }
            if (password.getText().length() == 0) { password.setError("请输入虚构的演示密码"); return; }
            password.setText("");
            account.setText("");
            hideKeyboard();
            entered = true;
            showHome();
        });
        addButton(form, "游客体验 · 无需账号", false, v -> {
            password.setText("");
            account.setText("");
            hideKeyboard();
            entered = true;
            showHome();
        });
        text(root, "当前未接入高德地图、真实定位和真实导航。所有地点均为模拟数据，不能用于实际出行。", textSize() - 2, muted, false);
        addButton(root, "无障碍显示设置", false, v -> showSettings());
        root.setFocusableInTouchMode(true);
        root.requestFocus();
    }

    private void showHome() {
        begin("home");
        heading("无障碍导航", "地图首页 · 本地演示模式");
        LinearLayout searchCard = card(root);
        searchInput = input(searchCard, "目的地", query, InputType.TYPE_CLASS_TEXT);
        searchInput.setHint("试试：公园、医院、服务中心");
        searchInput.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH);
        watch(searchInput, value -> query = value);
        addButton(searchCard, "搜索目的地", true, v -> doSearch());
        searchInput.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH) { doSearch(); return true; }
            return false;
        });
        text(searchCard, "仅搜索本机示例地点；匹配后将示意图移到结果位置。", textSize() - 3, muted, false);

        LinearLayout mapCard = card(root);
        text(mapCard, "地图模块 · 高德待接入", textSize(), ink, true);
        text(mapCard, "示意图，非真实地图。点击标记可勾选或取消；点击空白可添加自选点。", textSize() - 3, muted, false);
        mapModuleContainer = new FrameLayout(this);
        mapModuleContainer.setContentDescription("高德地图预留区域，目前显示非真实示意图");
        mapCard.addView(mapModuleContainer, new LinearLayout.LayoutParams(-1, dp(300)));
        demoMap = new DemoMapView();
        mapModuleContainer.addView(demoMap, new FrameLayout.LayoutParams(-1, -1));
        demoMap.centerOn(find(focusedId));
        focusStatus = text(mapCard, "", textSize() - 2, ink, true);
        focusStatus.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        addButton(mapCard, "查看全部示意地点", false, v -> demoMap.resetView());
        addButton(mapCard, "我的位置（待接入）", false, v -> notifyUser("尚未接入定位服务，无法获取你的真实位置。"));

        LinearLayout listCard = card(root);
        text(listCard, "地点选择", 23, ink, true);
        selectionStatus = text(listCard, "", textSize() - 2, muted, false);
        selectionStatus.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        resultsStatus = text(listCard, "", textSize() - 2, muted, false);
        results = column();
        listCard.addView(results, new LinearLayout.LayoutParams(-1, -2));
        addButton(listCard, "显示全部地点", false, v -> {
            searchInput.setText("");
            hideKeyboard();
            renderResults();
            demoMap.resetView();
        });
        addButton(listCard, "清空勾选", false, v -> { selected.clear(); renderResults(); updateMapStatus(); });
        addButton(listCard, "查看已勾选地点", true, v -> showSelected());
        text(listCard, "勾选用于收藏本次演示的地点，不会生成真实路线或优化多点出行顺序。", textSize() - 3, muted, false);
        addButton(root, "无障碍设置", false, v -> showSettings());
        addButton(root, "记录道路障碍", false, v -> showFeedback());
        addButton(root, "打开系统拨号盘联系帮助", false, v -> {
            try { startActivity(new Intent(Intent.ACTION_DIAL)); }
            catch (ActivityNotFoundException e) { notifyUser("此设备没有可用的拨号应用。"); }
        });
        addButton(root, "退出体验，返回登录", false, v -> {
            if (speech != null) speech.stop();
            entered = false;
            showLogin();
        });
        renderResults();
        updateMapStatus();
        root.setFocusableInTouchMode(true);
        root.requestFocus();
    }

    private List<Place> matching() {
        String keyword = query.trim().toLowerCase(Locale.ROOT);
        List<Place> found = new ArrayList<>();
        for (Place p : places) if (p.name.toLowerCase(Locale.ROOT).contains(keyword)) found.add(p);
        return found;
    }

    private void doSearch() {
        hideKeyboard();
        List<Place> found = matching();
        renderResults();
        if (found.isEmpty()) {
            notifyUser("没有匹配的示例地点。真实地点搜索需接入高德服务。");
        } else {
            focus(found.get(0), false);
            notifyUser(found.size() == 1 ? "已在示意图中显示搜索结果。" : "已显示第一项匹配，可在下方选择其他结果。");
        }
    }

    private void renderResults() {
        results.removeAllViews();
        List<Place> found = matching();
        resultsStatus.setText(found.isEmpty() ? "没有匹配结果，可点击“显示全部地点”。" : "找到 " + found.size() + " 个示例地点");
        for (Place p : found) {
            LinearLayout item = column();
            item.setPadding(0, dp(10), 0, dp(12));
            results.addView(item, new LinearLayout.LayoutParams(-1, -2));
            CheckBox check = new CheckBox(this);
            check.setText(p.name);
            check.setTextColor(ink);
            check.setTextSize(textSize());
            check.setMinHeight(dp(48));
            check.setButtonTintList(ColorStateList.valueOf(accent));
            check.setChecked(selected.contains(p.id));
            item.addView(check, new LinearLayout.LayoutParams(-1, -2));
            check.setOnCheckedChangeListener((b, checked) -> {
                if (checked) selected.add(p.id); else selected.remove(p.id);
                updateMapStatus();
            });
            text(item, p.description, textSize() - 3, muted, false);
            addButton(item, "在示意图中查看", false, v -> {
                focus(p, true);
                renderResults();
                notifyUser("已选中并显示：" + p.name);
            }).setContentDescription("在示意图中查看并勾选" + p.name);
        }
        updateMapStatus();
    }

    private void focus(Place p, boolean select) {
        if (p == null) return;
        focusedId = p.id;
        if (select) selected.add(p.id);
        if (demoMap != null) demoMap.centerOn(p);
        updateMapStatus();
    }

    private void updateMapStatus() {
        if (!"home".equals(page)) return;
        if (selectionStatus != null) selectionStatus.setText("已勾选 " + selected.size() + " 个地点 · 可多选");
        Place p = find(focusedId);
        if (focusStatus != null) focusStatus.setText(p == null ? "请选择示例地点" : "当前查看：" + p.name + "（模拟位置）");
        if (demoMap != null) demoMap.invalidate();
    }

    private void showSelected() {
        if (selected.isEmpty()) { notifyUser("请先勾选至少一个地点。"); return; }
        List<Place> items = new ArrayList<>();
        for (String id : selected) { Place p = find(id); if (p != null) items.add(p); }
        String[] names = new String[items.size()];
        for (int i = 0; i < items.size(); i++) names[i] = items.get(i).name;
        new AlertDialog.Builder(this).setTitle("已选地点 · 点击查看演示详情")
                .setItems(names, (dialog, which) -> {
                    focusedId = items.get(which).id;
                    routeStep = 0;
                    showRoute();
                }).setNegativeButton("关闭", null).show();
    }

    private void onMapTap(float x, float y) {
        if (demoMap == null) return;
        Place hit = demoMap.hit(x, y);
        if (hit != null) {
            focusedId = hit.id;
            boolean checked = !selected.contains(hit.id);
            if (checked) selected.add(hit.id); else selected.remove(hit.id);
            renderResults();
            notifyUser((checked ? "已勾选：" : "已取消勾选：") + hit.name);
            return;
        }
        float px = demoMap.worldX(x), py = demoMap.worldY(y);
        if (px < .05f || px > .95f || py < .05f || py > .95f) {
            notifyUser("请在示意区域内选点，或先点击查看全部示意地点。");
            return;
        }
        int count = 0;
        for (Place p : places) if (p.custom) count++;
        if (count >= 20) { notifyUser("本次体验最多添加 20 个自选点，可在设置中清除。"); return; }
        customCounter++;
        Place p = new Place("custom_" + customCounter, "自选点 " + customCounter,
                "手动添加的示意位置，无真实经纬度和设施信息", px, py, true);
        places.add(p);
        selected.add(p.id);
        focusedId = p.id;
        searchInput.setText("");
        renderResults();
        notifyUser("已添加并勾选：" + p.name);
    }

    private void showRoute() {
        Place p = find(focusedId);
        if (p == null) { showHome(); return; }
        begin("route");
        heading("地点与演示路线", "仅为界面演示，不能用于实际出行");
        LinearLayout box = card(root);
        text(box, p.name, 24, ink, true);
        text(box, p.description + "。设施信息未核验。", textSize(), muted, false);
        if (p.custom) {
            text(box, "自选点没有真实地理坐标，也没有演示路线。接入地图服务后才能按真实位置查询。", textSize(), ink, false);
        } else if ("bridge".equals(p.id) && avoidSteps) {
            text(box, "此模拟地点包含台阶，与“避开台阶”偏好冲突，不展示路线步骤。", textSize(), ink, true);
        } else {
            String[] steps = {
                    "从模拟社区入口出发。",
                    "沿示意通道前往“" + p.name + "”；此描述不对应真实道路。",
                    "演示结束。这不表示你已实际到达目的地。"
            };
            routeStep = Math.max(0, Math.min(routeStep, steps.length - 1));
            String instruction = "演示步骤 " + (routeStep + 1) + " / " + steps.length + "：" + steps[routeStep];
            text(box, instruction, textSize(), ink, true);
            addButton(box, "语音播报当前演示步骤", true, v -> speak("以下为模拟路线，不能用于实际出行。" + instruction));
            addButton(box, "停止播报", false, v -> { if (speech != null) speech.stop(); });
            addButton(box, "上一步", false, v -> {
                if (routeStep > 0) { routeStep--; stopSpeech(); showRoute(); }
                else notifyUser("已经是第一步。");
            });
            addButton(box, "下一步（手动演示）", false, v -> {
                if (routeStep < steps.length - 1) { routeStep++; stopSpeech(); showRoute(); }
                else notifyUser("演示已结束，不代表实际到达。");
            });
        }
        addButton(root, "记录此地点障碍", false, v -> { locationDraft = p.name; stopSpeech(); showFeedback(); });
        addButton(root, "返回地图首页", true, v -> { stopSpeech(); showHome(); });
    }

    private void toggle(LinearLayout parent, String label, boolean checked, android.widget.CompoundButton.OnCheckedChangeListener listener) {
        Switch s = new Switch(this);
        s.setText(label);
        s.setTextColor(ink);
        s.setTextSize(textSize());
        s.setMinHeight(dp(64));
        s.setChecked(checked);
        parent.addView(s, new LinearLayout.LayoutParams(-1, -2));
        s.setOnCheckedChangeListener(listener);
    }

    private void showSettings() {
        begin("settings");
        heading("无障碍设置", "显示偏好自动保存，返回时应用");
        LinearLayout box = card(root);
        toggle(box, "使用大字", largeText, (b, on) -> {
            largeText = on; prefs.edit().putBoolean("large_text", on).apply();
        });
        toggle(box, "高对比度", highContrast, (b, on) -> {
            highContrast = on; prefs.edit().putBoolean("map_high_contrast", on).apply();
        });
        toggle(box, "路线偏好：避开台阶", avoidSteps, (b, on) -> {
            avoidSteps = on; prefs.edit().putBoolean("avoid_steps", on).apply();
        });
        text(box, "避开台阶仅影响本机演示路线，不代表真实轮椅通行能力。坡度、路宽、电梯和临时障碍均需另行核验。", textSize() - 2, muted, false);
        text(box, "列表支持 TalkBack 勾选和查看地点。地图空白选点需要触摸，仅用于原型演示。", textSize() - 2, muted, false);
        addButton(root, "清除手动添加的示意点", false, v -> new AlertDialog.Builder(this)
                .setTitle("清除全部自选点？").setMessage("不影响示例地点或已保存的道路障碍记录。")
                .setNegativeButton("取消", null)
                .setPositiveButton("清除", (dialog, which) -> {
                    for (int i = places.size() - 1; i >= 0; i--) {
                        if (places.get(i).custom) { selected.remove(places.get(i).id); places.remove(i); }
                    }
                    if (find(focusedId) == null) focusedId = "center";
                    notifyUser("自选点已清除。");
                }).show());
        addButton(root, entered ? "应用并返回首页" : "应用并返回登录", true, v -> { if (entered) showHome(); else showLogin(); });
    }

    private void showFeedback() {
        begin("feedback");
        heading("道路障碍记录", "仅保存到本机，不上报、不触发救援");
        LinearLayout box = card(root);
        text(box, "请勿填写身份证号等敏感信息。记录不是已核验的路况。", textSize() - 2, muted, false);
        EditText location = input(box, "地点或路段（必填）", locationDraft, InputType.TYPE_CLASS_TEXT);
        EditText description = input(box, "障碍描述（必填）", feedbackDraft,
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        watch(location, value -> locationDraft = value);
        watch(description, value -> feedbackDraft = value);
        TextView status = text(box, "当前有 " + feedbackRecords().length() + " 条本地记录。", textSize(), ink, false);
        status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        addButton(box, "保存本地记录", true, v -> {
            if (locationDraft.trim().isEmpty() || feedbackDraft.trim().isEmpty()) {
                status.setText("保存失败：请填写地点和障碍描述。"); return;
            }
            if (locationDraft.length() > 200 || feedbackDraft.length() > 2000) {
                status.setText("保存失败：地点不能超过 200 字，描述不能超过 2000 字。"); return;
            }
            try {
                JSONArray records = feedbackRecords();
                if (records.length() >= 100) { status.setText("最多保存 100 条记录，请先清空旧记录。"); return; }
                JSONObject item = new JSONObject();
                item.put("location", locationDraft.trim());
                item.put("description", feedbackDraft.trim());
                item.put("createdAt", System.currentTimeMillis());
                records.put(item);
                prefs.edit().putString("feedback_records", records.toString()).apply();
                description.setText("");
                hideKeyboard();
                status.setText("已保存到本机，共 " + records.length() + " 条；尚未上报。");
            } catch (JSONException e) { status.setText("保存失败，请重试。"); }
        });
        addButton(box, "查看本地记录", false, v -> {
            JSONArray records = feedbackRecords();
            StringBuilder lines = new StringBuilder();
            for (int i = 0; i < records.length(); i++) {
                JSONObject item = records.optJSONObject(i);
                if (item != null) lines.append(i + 1).append(". ").append(item.optString("location"))
                        .append("\n").append(item.optString("description")).append("\n\n");
            }
            status.setText(lines.length() == 0 ? "暂无本地记录。" : lines.toString());
        });
        addButton(box, "清空本地记录", false, v -> new AlertDialog.Builder(this)
                .setTitle("清空全部本地记录？").setMessage("删除后无法恢复。")
                .setNegativeButton("取消", null)
                .setPositiveButton("确认清空", (dialog, which) -> {
                    prefs.edit().remove("feedback_records").apply();
                    status.setText("本地记录已清空。");
                }).show());
        addButton(root, "返回首页", false, v -> { hideKeyboard(); showHome(); });
    }

    private JSONArray feedbackRecords() {
        try { return new JSONArray(prefs.getString("feedback_records", "[]")); }
        catch (JSONException e) { return new JSONArray(); }
    }

    private void speak(String message) {
        if (!speechReady || speech == null) { notifyUser("中文语音尚未就绪或未安装，请使用文字提示。"); return; }
        if (speech.speak(message, TextToSpeech.QUEUE_FLUSH, null, "demo_step") == TextToSpeech.ERROR)
            notifyUser("播报失败，请查看文字提示。");
    }
    private void stopSpeech() { if (speech != null) speech.stop(); }
    private void notifyUser(String message) { Toast.makeText(this, message, Toast.LENGTH_LONG).show(); }
    private Place find(String id) { for (Place p : places) if (p.id.equals(id)) return p; return null; }

    @Override protected void onSaveInstanceState(Bundle out) {
        out.putBoolean("entered", entered);
        out.putString("page", page);
        out.putString("query", query);
        out.putString("focus", focusedId);
        out.putString("feedback", feedbackDraft);
        out.putString("location", locationDraft);
        out.putInt("step", routeStep);
        out.putInt("counter", customCounter);
        out.putStringArrayList("selected", new ArrayList<>(selected));
        JSONArray custom = new JSONArray();
        for (Place p : places) if (p.custom) {
            try {
                JSONObject item = new JSONObject();
                item.put("id", p.id); item.put("name", p.name);
                item.put("x", p.x); item.put("y", p.y);
                custom.put(item);
            } catch (JSONException ignored) { /* 本机有限坐标不包含非法数值。 */ }
        }
        out.putString("custom", custom.toString());
        super.onSaveInstanceState(out);
    }

    private void restoreCustomPlaces(String json) {
        try {
            JSONArray array = new JSONArray(json);
            for (int i = 0; i < array.length(); i++) {
                JSONObject item = array.getJSONObject(i);
                places.add(new Place(item.getString("id"), item.getString("name"),
                        "手动添加的示意位置，无真实经纬度和设施信息",
                        (float) item.getDouble("x"), (float) item.getDouble("y"), true));
            }
        } catch (JSONException ignored) { /* 无法读取的临时示意点不恢复。 */ }
    }

    @Override public void onBackPressed() {
        stopSpeech();
        if ("login".equals(page) || "home".equals(page)) super.onBackPressed();
        else if (entered) showHome();
        else showLogin();
    }
    @Override protected void onStop() { stopSpeech(); super.onStop(); }
    @Override protected void onDestroy() {
        if (speech != null) { speech.stop(); speech.shutdown(); }
        super.onDestroy();
    }

    private static final class Place {
        final String id, name, description;
        final float x, y;
        final boolean custom;
        Place(String id, String name, String description, float x, float y, boolean custom) {
            this.id = id; this.name = name; this.description = description;
            this.x = x; this.y = y; this.custom = custom;
        }
    }

    /** 可运行的示意图组件。坐标为 0～1 的画布位置，不是经纬度。 */
    private final class DemoMapView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private float centerX = .5f, centerY = .5f;
        private float downX, downY;
        DemoMapView() {
            super(MainActivity.this);
            setBackground(rounded(highContrast ? Color.rgb(30, 35, 40) : Color.rgb(231, 239, 233), 12));
            setClipToOutline(true);
            setClickable(true);
            // 所有已有地点的可访问操作都在下方原生复选框和按钮中提供。
            setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        }
        void centerOn(Place p) {
            if (p != null) { centerX = p.x; centerY = p.y; }
            invalidate();
        }
        void resetView() { centerX = .5f; centerY = .5f; invalidate(); }
        float sx(float x) { return (x - centerX + .5f) * getWidth(); }
        float sy(float y) { return (y - centerY + .5f) * getHeight(); }
        float worldX(float x) { return x / Math.max(1, getWidth()) + centerX - .5f; }
        float worldY(float y) { return y / Math.max(1, getHeight()) + centerY - .5f; }
        Place hit(float x, float y) {
            Place best = null;
            double min = dp(25);
            for (Place p : places) {
                double distance = Math.hypot(sx(p.x) - x, sy(p.y) - y);
                if (distance < min) { min = distance; best = p; }
            }
            return best;
        }
        private void road(Canvas c, float x1, float y1, float x2, float y2, int color, float width) {
            paint.setColor(color); paint.setStrokeWidth(dp(width)); paint.setStyle(Paint.Style.STROKE);
            c.drawLine(sx(x1), sy(y1), sx(x2), sy(y2), paint);
            paint.setStyle(Paint.Style.FILL);
        }
        @Override protected void onDraw(Canvas c) {
            super.onDraw(c);
            paint.setColor(highContrast ? Color.rgb(39, 61, 47) : Color.rgb(195, 223, 197));
            c.drawRoundRect(sx(.54f), sy(.08f), sx(.89f), sy(.41f), dp(22), dp(22), paint);
            paint.setColor(highContrast ? Color.rgb(41, 68, 82) : Color.rgb(171, 212, 231));
            Path river = new Path();
            river.moveTo(sx(.82f), sy(-.5f)); river.cubicTo(sx(.55f), sy(.4f), sx(1.1f), sy(.6f), sx(.87f), sy(1.5f));
            paint.setStyle(Paint.Style.STROKE); paint.setStrokeWidth(dp(23)); c.drawPath(river, paint);
            paint.setStyle(Paint.Style.FILL);
            int roadColor = highContrast ? Color.rgb(88, 96, 103) : Color.WHITE;
            road(c, -.5f, .46f, 1.5f, .46f, roadColor, 22);
            road(c, .44f, -.5f, .44f, 1.5f, roadColor, 19);
            road(c, -.5f, .84f, 1.5f, .84f, roadColor, 13);
            road(c, .12f, -.5f, .12f, 1.5f, roadColor, 12);
            paint.setTypeface(Typeface.DEFAULT);
            paint.setTextSize(dp(11)); paint.setTextAlign(Paint.Align.CENTER);
            paint.setColor(highContrast ? Color.LTGRAY : Color.rgb(88, 110, 128));
            c.drawText("示意通道", sx(.25f), sy(.46f) + dp(4), paint);
            for (Place p : places) {
                float x = sx(p.x), y = sy(p.y);
                boolean checked = selected.contains(p.id);
                if (p.id.equals(focusedId)) {
                    paint.setColor(highContrast ? 0x66FFE255 : 0x332563EB);
                    c.drawCircle(x, y, dp(23), paint);
                }
                paint.setColor(Color.WHITE); c.drawCircle(x, y, dp(15), paint);
                paint.setColor(checked ? accent : (highContrast ? Color.LTGRAY : Color.rgb(90, 112, 138)));
                c.drawCircle(x, y, dp(12), paint);
                paint.setTextSize(dp(14));
                paint.setTypeface(Typeface.DEFAULT_BOLD);
                paint.setColor(checked && highContrast ? Color.BLACK : Color.WHITE);
                c.drawText(checked ? "✓" : "·", x, y + dp(5), paint);
                paint.setTextSize(dp(11));
                float labelWidth = paint.measureText(p.name) + dp(12);
                float lx = Math.max(labelWidth / 2 + dp(2), Math.min(getWidth() - labelWidth / 2 - dp(2), x));
                float ly = y + dp(19);
                if (x >= 0 && x <= getWidth() && y >= 0 && y <= getHeight()) {
                    paint.setColor(highContrast ? Color.BLACK : Color.WHITE);
                    c.drawRoundRect(lx - labelWidth / 2, ly, lx + labelWidth / 2, ly + dp(20), dp(5), dp(5), paint);
                    paint.setColor(ink); c.drawText(p.name, lx, ly + dp(14), paint);
                }
            }
            paint.setColor(highContrast ? Color.BLACK : Color.WHITE);
            c.drawRoundRect(dp(8), dp(8), dp(170), dp(34), dp(6), dp(6), paint);
            paint.setTextSize(dp(12)); paint.setColor(ink); paint.setTextAlign(Paint.Align.LEFT);
            c.drawText("示意图 · 非真实地理位置", dp(15), dp(26), paint);
        }
        @Override public boolean onTouchEvent(MotionEvent e) {
            if (e.getActionMasked() == MotionEvent.ACTION_DOWN) {
                downX = e.getX(); downY = e.getY(); return true;
            }
            if (e.getActionMasked() == MotionEvent.ACTION_UP) {
                if (Math.hypot(e.getX() - downX, e.getY() - downY) < dp(12)) {
                    performClick();
                    onMapTap(e.getX(), e.getY());
                }
                return true;
            }
            return e.getActionMasked() == MotionEvent.ACTION_MOVE || e.getActionMasked() == MotionEvent.ACTION_CANCEL;
        }
        @Override public boolean performClick() { super.performClick(); return true; }
    }
}
