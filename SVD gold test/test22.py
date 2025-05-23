# Store results
self.hasil['harga_prediksi'] = harga_prediksi

# Calculate confidence interval if requested
if self.var_confidence.get():
    self.hasil['confidence_interval'] = self.hitung_confidence_interval(M, y, Mx)

# Calculate residuals for training data
if self.var_residual_analysis.get():
    self.hitung_residuals(M, y, model_type)

self.update_progress(90, "Memperbarui tampilan...")

# Update UI
self.label_hasil.config(text=f"${harga_prediksi:.2f} USD")
self.update_detail_results()
self.perbarui_visualisasi_prediksi()

self.update_progress(100, "Selesai!")

messagebox.showinfo("Prediksi Selesai",
                    f"Harga emas prediksi untuk tahun {tahun}: ${harga_prediksi:.2f} USD")

except ValueError as e:
messagebox.showerror("Kesalahan Input", f"Input tidak valid: {str(e)}")
except Exception as e:
messagebox.showerror("Kesalahan Perhitungan", f"Terjadi kesalahan: {str(e)}")


def prediksi_svd(self, M, y, Mx):
    """Prediksi menggunakan SVD"""
    # SVD decomposition
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    V = VT.T

    # Calculate U^T * y
    UT_y = U.T @ y

    # Calculate pseudo-inverse
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > 1e-10:  # Avoid division by very small numbers
            S_inv[i] = 1 / S[i]

    # Calculate omega
    omega = V @ (S_inv * UT_y)

    # Store SVD components
    self.hasil['komponen_svd'] = {'U': U, 'S': S, 'V': V}
    self.hasil['vektor_omega'] = omega

    # Update SVD display
    self.update_svd_display(U, S, V, omega)

    # Calculate training predictions for residuals
    y_train_pred = M @ omega
    self.hasil['y_train_pred'] = y_train_pred

    # Prediction
    return Mx @ omega


def prediksi_ml_model(self, M, y, Mx, model_type):
    """Prediksi menggunakan model ML"""
    try:
        # Create new instance to avoid state issues
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Fit the model
        model.fit(M, y)

        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.hasil['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            self.hasil['feature_importance'] = np.abs(model.coef_)

        # Calculate training predictions for residuals
        y_train_pred = model.predict(M)
        self.hasil['y_train_pred'] = y_train_pred

        # Make prediction
        prediction = model.predict(Mx.reshape(1, -1))[0]

        # Update model display
        self.update_model_display(model, model_type)

        return prediction

    except Exception as e:
        raise Exception(f"Error dalam model {model_type}: {str(e)}")


def hitung_residuals(self, M, y, model_type):
    """Calculate residuals for training data"""
    if 'y_train_pred' in self.hasil and self.hasil['y_train_pred'] is not None:
        residuals = y - self.hasil['y_train_pred']
        self.hasil['residuals'] = residuals
    else:
        self.hasil['residuals'] = None


def update_svd_display(self, U, S, V, omega):
    """Update SVD components display"""
    svd_info = f"""=== Komponen SVD ===

Matrix U (Left Singular Vectors):
{U}

Singular Values (S):
{S}

Matrix V (Right Singular Vectors):
{V}

Vektor Omega (Koefisien):
{omega}

=== Informasi Tambahan ===
Condition Number: {np.max(S) / np.min(S[S > 1e-10]):.2f}
Rank: {np.sum(S > 1e-10)}
"""

    self.teks_svd.config(state=tk.NORMAL)
    self.teks_svd.delete(1.0, tk.END)
    self.teks_svd.insert(tk.END, svd_info)
    self.teks_svd.config(state=tk.DISABLED)


def update_model_display(self, model, model_type):
    """Update ML model display"""
    model_info = f"=== Model: {model_type.upper()} ===\n\n"

    if hasattr(model, 'coef_'):
        model_info += f"Koefisien:\n{model.coef_}\n\n"

    if hasattr(model, 'intercept_'):
        model_info += f"Intercept: {model.intercept_:.4f}\n\n"

    if hasattr(model, 'feature_importances_'):
        model_info += f"Feature Importances:\n{model.feature_importances_}\n\n"

    # Calculate R² score for training data
    M = np.array([
        self.data_historis['inflasi'],
        self.data_historis['suku_bunga'],
        self.data_historis['indeks_usd']
    ]).T
    y = np.array(self.data_historis['harga_emas'])

    y_pred_train = model.predict(M)
    r2_train = r2_score(y, y_pred_train)
    model_info += f"R² Score (Training): {r2_train:.4f}\n"

    self.teks_svd.config(state=tk.NORMAL)
    self.teks_svd.delete(1.0, tk.END)
    self.teks_svd.insert(tk.END, model_info)
    self.teks_svd.config(state=tk.DISABLED)


def hitung_confidence_interval(self, M, y, Mx, confidence=0.95):
    """Hitung confidence interval untuk prediksi"""
    # Simplified bootstrap approach
    n_bootstrap = 100
    predictions = []

    for _ in range(n_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(len(y), size=len(y), replace=True)
        M_boot = M[indices]
        y_boot = y[indices]

        # Make prediction
        try:
            if self.model_terpilih.get() == "svd":
                pred = self.prediksi_svd(M_boot, y_boot, Mx)
            else:
                pred = self.prediksi_ml_model(M_boot, y_boot, Mx, self.model_terpilih.get())
            predictions.append(pred)
        except:
            continue

    if predictions:
        alpha = 1 - confidence
        lower = np.percentile(predictions, 100 * alpha / 2)
        upper = np.percentile(predictions, 100 * (1 - alpha / 2))
        return (lower, upper)

    return None


def update_detail_results(self):
    """Update detail results display"""
    self.teks_detail.config(state=tk.NORMAL)
    self.teks_detail.delete(1.0, tk.END)

    detail = f"""Hasil Prediksi Harga Emas

Tahun: {self.parameter_prediksi['tahun']}
Model: {self.model_terpilih.get().upper()}

Parameter Input:
- Inflasi: {self.parameter_prediksi['inflasi']}%
- Suku Bunga: {self.parameter_prediksi['suku_bunga']}%
- Indeks USD: {self.parameter_prediksi['indeks_usd']}

Harga Prediksi: ${self.hasil['harga_prediksi']:.4f} USD

"""

    if self.hasil['confidence_interval']:
        ci = self.hasil['confidence_interval']
        detail += f"""Confidence Interval (95%):
- Lower: ${ci[0]:.2f}
- Upper: ${ci[1]:.2f}

"""

    if self.hasil['feature_importance'] is not None:
        detail += "Feature Importance:\n"
        features = ['Inflasi', 'Suku Bunga', 'Indeks USD']
        for i, (feat, imp) in enumerate(zip(features, self.hasil['feature_importance'])):
            detail += f"- {feat}: {imp:.4f}\n"

    detail += f"\nWaktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    self.teks_detail.insert(tk.END, detail)
    self.teks_detail.config(state=tk.DISABLED)


def perbarui_visualisasi_prediksi(self, event=None):
    """Update prediction visualization"""
    if self.hasil['harga_prediksi'] is None:
        return

    self.gambar_pred.clear()

    view_type = self.pred_view.get()

    if view_type == "trend":
        self.plot_trend_prediction()
    elif view_type == "confidence":
        self.plot_confidence_prediction()
    elif view_type == "residuals":
        self.plot_residuals()
    elif view_type == "comparison":
        self.plot_model_comparison()

    self.gambar_pred.tight_layout()
    self.kanvas_pred.draw()


def plot_trend_prediction(self):
    """Plot trend with prediction"""
    ax = self.gambar_pred.add_subplot(111)

    # Historical data
    tahun_hist = self.data_historis['tahun']
    harga_hist = self.data_historis['harga_emas']

    # Plot historical
    ax.plot(tahun_hist, harga_hist, 'o-', color='blue', linewidth=2,
            label='Data Historis', markersize=8)

    # Plot prediction
    tahun_pred = self.parameter_prediksi['tahun']
    harga_pred = self.hasil['harga_prediksi']

    ax.plot([tahun_hist[-1], tahun_pred], [harga_hist[-1], harga_pred],
            'o--', color='red', linewidth=2, label='Prediksi', markersize=8)

    # Highlight prediction point
    ax.scatter([tahun_pred], [harga_pred], color='red', s=150, zorder=5)

    # Add annotation
    ax.annotate(f"${harga_pred:.2f}",
                xy=(tahun_pred, harga_pred),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

    ax.set_title(f'Prediksi Harga Emas - {tahun_pred}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Harga Emas (USD)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()


def plot_confidence_prediction(self):
    """Plot prediction with confidence interval"""
    if not self.hasil['confidence_interval']:
        self.plot_trend_prediction()
        return

    ax = self.gambar_pred.add_subplot(111)

    # Historical data
    tahun_hist = self.data_historis['tahun']
    harga_hist = self.data_historis['harga_emas']

    ax.plot(tahun_hist, harga_hist, 'o-', color='blue', linewidth=2,
            label='Data Historis')

    # Prediction with confidence interval
    tahun_pred = self.parameter_prediksi['tahun']
    harga_pred = self.hasil['harga_prediksi']
    ci_lower, ci_upper = self.hasil['confidence_interval']

    ax.errorbar([tahun_pred], [harga_pred],
                yerr=[[harga_pred - ci_lower], [ci_upper - harga_pred]],
                fmt='ro', capsize=10, capthick=2, markersize=10,
                label=f'Prediksi (CI 95%)')

    ax.set_title('Prediksi dengan Confidence Interval', fontsize=14)
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Harga Emas (USD)')
    ax.grid(True, alpha=0.7)
    ax.legend()


def plot_residuals(self):
    """Plot residuals analysis"""
    if self.hasil['residuals'] is None:
        ax = self.gambar_pred.add_subplot(111)
        ax.text(0.5, 0.5, 'Residuals not available\nRun prediction first',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Residuals Analysis')
        return

    # Create 2x2 subplot layout for residuals analysis
    ax1 = self.gambar_pred.add_subplot(221)
    ax2 = self.gambar_pred.add_subplot(222)
    ax3 = self.gambar_pred.add_subplot(223)
    ax4 = self.gambar_pred.add_subplot(224)

    residuals = self.hasil['residuals']
    y_actual = np.array(self.data_historis['harga_emas'])
    y_pred = self.hasil['y_train_pred']
    tahun = self.data_historis['tahun']

    # 1. Residuals vs Fitted
    ax1.scatter(y_pred, residuals, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)

    # 2. Normal Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Residuals vs Time (Years)
    ax3.plot(tahun, residuals, 'o-', alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Time')
    ax3.grid(True, alpha=0.3)

    # 4. Histogram of residuals
    ax4.hist(residuals, bins=5, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Histogram of Residuals')
    ax4.grid(True, alpha=0.3)

    # Add statistics text
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax4.text(0.05, 0.95, f'Mean: {mean_res:.2f}\nStd: {std_res:.2f}',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_model_comparison(self):
    """Plot comparison of different models"""
    # Run all models for comparison
    M = np.array([
        self.data_historis['inflasi'],
        self.data_historis['suku_bunga'],
        self.data_historis['indeks_usd']
    ]).T
    y = np.array(self.data_historis['harga_emas'])
    Mx = np.array([
        self.parameter_prediksi['inflasi'],
        self.parameter_prediksi['suku_bunga'],
        self.parameter_prediksi['indeks_usd']
    ])

    models = ['svd', 'linear', 'ridge', 'lasso', 'random_forest']
    model_names = ['SVD', 'Linear', 'Ridge', 'Lasso', 'Random Forest']
    predictions = []
    r2_scores = []
    rmse_scores = []

    for model_type in models:
        try:
            if model_type == 'svd':
                # SVD prediction
                U, S, VT = np.linalg.svd(M, full_matrices=False)
                V = VT.T
                UT_y = U.T @ y
                S_inv = np.zeros_like(S)
                for i in range(len(S)):
                    if S[i] > 1e-10:
                        S_inv[i] = 1 / S[i]
                omega = V @ (S_inv * UT_y)
                pred = Mx @ omega
                y_train_pred = M @ omega
            else:
                # ML models
                if model_type == 'linear':
                    model = LinearRegression()
                elif model_type == 'ridge':
                    model = Ridge(alpha=1.0)
                elif model_type == 'lasso':
                    model = Lasso(alpha=1.0)
                elif model_type == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(M, y)
                pred = model.predict(Mx.reshape(1, -1))[0]
                y_train_pred = model.predict(M)

            predictions.append(pred)
            r2_scores.append(r2_score(y, y_train_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y, y_train_pred)))

        except Exception as e:
            predictions.append(0)
            r2_scores.append(0)
            rmse_scores.append(0)

    # Create comparison plots
    ax1 = self.gambar_pred.add_subplot(221)
    ax2 = self.gambar_pred.add_subplot(222)
    ax3 = self.gambar_pred.add_subplot(223)
    ax4 = self.gambar_pred.add_subplot(224)

    # 1. Predictions comparison
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    bars1 = ax1.bar(model_names, predictions, color=colors, alpha=0.7)
    ax1.set_title('Model Predictions Comparison')
    ax1.set_ylabel('Predicted Price (USD)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, pred in zip(bars1, predictions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'${pred:.0f}', ha='center', va='bottom', fontsize=8)

    # 2. R² scores comparison
    bars2 = ax2.bar(model_names, r2_scores, color=colors, alpha=0.7)
    ax2.set_title('R² Score Comparison')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add value labels
    for bar, r2 in zip(bars2, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. RMSE comparison
    bars3 = ax3.bar(model_names, rmse_scores, color=colors, alpha=0.7)
    ax3.set_title('RMSE Comparison')
    ax3.set_ylabel('RMSE')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, rmse in zip(bars3, rmse_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{rmse:.1f}', ha='center', va='bottom', fontsize=8)

    # 4. Summary table
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    for i, name in enumerate(model_names):
        table_data.append([name, f'${predictions[i]:.0f}', f'{r2_scores[i]:.3f}', f'{rmse_scores[i]:.1f}'])

    table = ax4.table(cellText=table_data,
                      colLabels=['Model', 'Prediction', 'R²', 'RMSE'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.3, 0.25, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax4.set_title('Model Performance Summary')

    # === Advanced Analysis Methods ===


def jalankan_analisis_lanjutan(self):
    """Jalankan analisis lanjutan komprehensif"""
    try:
        self.update_progress(5, "Memulai analisis lanjutan...")

        # Clear previous results
        self.clear_analisis_results()

        # Get data
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T
        y = np.array(self.data_historis['harga_emas'])
        years = np.array(self.data_historis['tahun'])

        analysis_results = {}

        # 1. Correlation Analysis
        if self.analisis_vars['correlation'].get():
            self.update_progress(15, "Analisis korelasi...")
            analysis_results['correlation'] = self.analisis_korelasi(M, y)
            self.plot_correlation_analysis(analysis_results['correlation'])

        # 2. Sensitivity Analysis
        if self.analisis_vars['sensitivity'].get():
            self.update_progress(25, "Analisis sensitivitas...")
            analysis_results['sensitivity'] = self.analisis_sensitivitas(M, y)
            self.plot_sensitivity_analysis(analysis_results['sensitivity'])

        # 3. Trend Analysis
        if self.analisis_vars['trend'].get():
            self.update_progress(35, "Analisis tren...")
            analysis_results['trend'] = self.analisis_tren(y, years)

        # 4. Volatility Analysis
        if self.analisis_vars['volatility'].get():
            self.update_progress(45, "Analisis volatilitas...")
            analysis_results['volatility'] = self.analisis_volatilitas(y)

        # Plot trend and volatility together
        if self.analisis_vars['trend'].get() or self.analisis_vars['volatility'].get():
            self.plot_trend_volatility_analysis(analysis_results.get('trend'),
                                                analysis_results.get('volatility'), years, y)

        # 5. Multi-step Forecast
        if self.analisis_vars['forecast'].get():
            self.update_progress(65, "Peramalan multi-step...")
            periods = int(self.forecast_periods.get())
            analysis_results['forecast'] = self.analisis_forecast(M, y, years, periods)
            self.plot_forecast_analysis(analysis_results['forecast'], years, y)
            self.update_forecast_table(analysis_results['forecast'])

        # 6. Risk Analysis
        if self.analisis_vars['risk'].get():
            self.update_progress(75, "Analisis risiko...")
            analysis_results['risk'] = self.analisis_risiko(y, analysis_results.get('forecast'))
            self.plot_risk_analysis(analysis_results['risk'], y)

        # 7. Generate Summary
        self.update_progress(95, "Membuat ringkasan...")
        self.generate_analysis_summary(analysis_results)

        self.update_progress(100, "Analisis selesai!")
        messagebox.showinfo("Analisis Selesai", "Analisis lanjutan telah selesai. Periksa tab-tab untuk melihat hasil.")

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan dalam analisis: {str(e)}")
    finally:
        self.reset_progress()


def analisis_korelasi(self, M, y):
    """Analisis korelasi antar variabel"""
    # Create correlation matrix
    all_data = np.column_stack([M, y])
    feature_names = ['Inflasi', 'Suku Bunga', 'Indeks USD', 'Harga Emas']

    corr_matrix = np.corrcoef(all_data.T)

    # Calculate partial correlations
    partial_corr = {}
    for i in range(3):  # For each feature vs gold price
        # Simple partial correlation (controlling for other variables)
        other_features = [j for j in range(3) if j != i]
        X_control = M[:, other_features]

        # Residuals after controlling for other variables
        reg_x = LinearRegression().fit(X_control, M[:, i])
        reg_y = LinearRegression().fit(X_control, y)

        resid_x = M[:, i] - reg_x.predict(X_control)
        resid_y = y - reg_y.predict(X_control)

        partial_corr[feature_names[i]] = np.corrcoef(resid_x, resid_y)[0, 1]

    return {
        'correlation_matrix': corr_matrix,
        'feature_names': feature_names,
        'partial_correlations': partial_corr
    }


def analisis_sensitivitas(self, M, y):
    """Analisis sensitivitas parameter"""
    # Sensitivity ranges (percentage changes)
    sensitivity_ranges = np.arange(-20, 21, 5)  # -20% to +20% in 5% steps
    base_values = np.mean(M, axis=0)

    sensitivity_results = {}
    feature_names = ['Inflasi', 'Suku Bunga', 'Indeks USD']

    # Train base model
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    V = VT.T
    UT_y = U.T @ y
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > 1e-10:
            S_inv[i] = 1 / S[i]
    omega = V @ (S_inv * UT_y)

    base_prediction = base_values @ omega

    for i, feature_name in enumerate(feature_names):
        predictions = []
        for pct_change in sensitivity_ranges:
            test_values = base_values.copy()
            test_values[i] *= (1 + pct_change / 100)
            pred = test_values @ omega
            predictions.append(pred)

        sensitivity_results[feature_name] = {
            'ranges': sensitivity_ranges,
            'predictions': np.array(predictions),
            'base_prediction': base_prediction,
            'elasticity': np.gradient(predictions, sensitivity_ranges)
        }

    return sensitivity_results


def analisis_tren(self, y, years):
    """Analisis tren harga emas"""
    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, y)

    # Polynomial trends
    poly_2 = np.polyfit(years, y, 2)
    poly_3 = np.polyfit(years, y, 3)

    # Trend strength
    y_trend = slope * years + intercept
    trend_strength = r_value ** 2

    # Growth rate
    annual_growth_rate = slope / np.mean(y) * 100

    # Trend classification
    if slope > 0:
        trend_direction = "Naik" if p_value < 0.05 else "Naik (tidak signifikan)"
    else:
        trend_direction = "Turun" if p_value < 0.05 else "Turun (tidak signifikan)"

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'trend_line': y_trend,
        'poly_2': poly_2,
        'poly_3': poly_3,
        'trend_strength': trend_strength,
        'annual_growth_rate': annual_growth_rate,
        'trend_direction': trend_direction
    }


def analisis_volatilitas(self, y):
    """Analisis volatilitas harga emas"""
    window = int(self.volatility_window.get())

    # Returns
    returns = np.diff(y) / y[:-1] * 100

    # Rolling volatility
    if len(returns) >= window:
        rolling_vol = []
        for i in range(window - 1, len(returns)):
            window_data = returns[i - window + 1:i + 1]
            rolling_vol.append(np.std(window_data))
        rolling_vol = np.array(rolling_vol)
    else:
        rolling_vol = np.array([np.std(returns)])

    # Volatility statistics
    vol_stats = self.teks_detail = tk.Text(tab_detail, height=8, width=60)
    detail_scroll = ttk.Scrollbar(tab_detail, orient=tk.VERTICAL, command=self.teks_detail.yview)
    self.teks_detail.configure(yscrollcommand=detail_scroll.set)

    self.teks_detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # Tab metrics
    tab_metrics = ttk.Frame(detail_notebook)
    detail_notebook.add(tab_metrics, text="Metrics")

    self.teks_metrics = tk.Text(tab_metrics, height=8, width=60)
    metrics_scroll = ttk.Scrollbar(tab_metrics, orient=tk.VERTICAL, command=self.teks_metrics.yview)
    self.teks_metrics.configure(yscrollcommand=metrics_scroll.set)

    self.teks_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # Visualisasi prediksi
    viz_pred_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Prediksi")
    viz_pred_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

    # Kontrol visualisasi prediksi
    pred_control = ttk.Frame(viz_pred_frame)
    pred_control.pack(fill=tk.X, pady=2)

    ttk.Label(pred_control, text="Tampilan:").pack(side=tk.LEFT, padx=5)
    self.pred_view = tk.StringVar(value="trend")
    combo_pred = ttk.Combobox(pred_control, textvariable=self.pred_view,
                              values=["trend", "confidence", "residuals", "comparison"],
                              width=12, state="readonly")
    combo_pred.pack(side=tk.LEFT, padx=5)
    combo_pred.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_prediksi)

    ttk.Button(pred_control, text="🔄 Perbarui",
               command=self.perbarui_visualisasi_prediksi).pack(side=tk.LEFT, padx=5)

    self.gambar_pred = Figure(figsize=(10, 6), dpi=100)
    self.kanvas_pred = FigureCanvasTkAgg(self.gambar_pred, viz_pred_frame)

    # Toolbar navigasi
    toolbar_pred_frame = ttk.Frame(viz_pred_frame)
    toolbar_pred_frame.pack(fill=tk.X, padx=5, pady=2)

    self.navbar_pred = NavigationToolbar2Tk(self.kanvas_pred, toolbar_pred_frame)
    self.navbar_pred.update()

    self.kanvas_pred.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def buat_tab_analisis(self):
    """Membuat tab analisis lanjutan"""
    bingkai = ttk.Frame(self.tab_analisis)
    bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # PanedWindow untuk split layout
    paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    # Frame kiri - Controls dan opsi analisis
    frame_kiri = ttk.Frame(paned)
    paned.add(frame_kiri, weight=1)

    # Frame kanan - Hasil analisis dan visualisasi
    frame_kanan = ttk.Frame(paned)
    paned.add(frame_kanan, weight=3)

    # === Frame Kiri ===
    # Kontrol Analisis
    control_frame = ttk.LabelFrame(frame_kiri, text="Kontrol Analisis")
    control_frame.pack(fill=tk.X, pady=5, padx=5)

    # Jenis analisis
    ttk.Label(control_frame, text="Jenis Analisis:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=2)

    self.analisis_vars = {}
    analisis_options = [
        ("correlation", "Analisis Korelasi"),
        ("sensitivity", "Analisis Sensitivitas"),
        ("trend", "Analisis Tren"),
        ("volatility", "Analisis Volatilitas"),
        ("seasonality", "Analisis Musiman"),
        ("forecast", "Peramalan Multi-Step"),
        ("risk", "Analisis Risiko"),
        ("feature_impact", "Dampak Fitur")
    ]

    for key, label in analisis_options:
        var = tk.BooleanVar(value=True)
        self.analisis_vars[key] = var
        ttk.Checkbutton(control_frame, text=label, variable=var).pack(anchor=tk.W, padx=10, pady=1)

    # Parameter analisis
    param_frame = ttk.LabelFrame(frame_kiri, text="Parameter Analisis")
    param_frame.pack(fill=tk.X, pady=5, padx=5)

    # Periode peramalan
    ttk.Label(param_frame, text="Periode Forecast:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    self.forecast_periods = tk.StringVar(value="3")
    ttk.Entry(param_frame, textvariable=self.forecast_periods, width=10).grid(row=0, column=1, padx=5, pady=2)

    # Confidence level
    ttk.Label(param_frame, text="Confidence Level (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    self.confidence_level = tk.StringVar(value="95")
    ttk.Entry(param_frame, textvariable=self.confidence_level, width=10).grid(row=1, column=1, padx=5, pady=2)

    # Monte Carlo simulations
    ttk.Label(param_frame, text="MC Simulations:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
    self.mc_simulations = tk.StringVar(value="1000")
    ttk.Entry(param_frame, textvariable=self.mc_simulations, width=10).grid(row=2, column=1, padx=5, pady=2)

    # Volatility window
    ttk.Label(param_frame, text="Volatility Window:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
    self.volatility_window = tk.StringVar(value="2")
    ttk.Entry(param_frame, textvariable=self.volatility_window, width=10).grid(row=3, column=1, padx=5, pady=2)

    # Tombol eksekusi
    btn_frame = ttk.Frame(param_frame)
    btn_frame.grid(row=4, column=0, columnspan=2, pady=10)

    ttk.Button(btn_frame, text="🚀 Jalankan Analisis",
               command=self.jalankan_analisis_lanjutan).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="📊 Reset",
               command=self.reset_analisis).pack(side=tk.LEFT, padx=2)

    # Scenario Analysis
    scenario_frame = ttk.LabelFrame(frame_kiri, text="Analisis Skenario")
    scenario_frame.pack(fill=tk.X, pady=5, padx=5)

    # Skenario ekonomi
    scenarios = [
        ("optimistic", "Optimistis", {"inflasi": 2.0, "suku_bunga": 4.0, "indeks_usd": 100.0}),
        ("moderate", "Moderat", {"inflasi": 3.5, "suku_bunga": 5.5, "indeks_usd": 105.0}),
        ("pessimistic", "Pesimistis", {"inflasi": 5.0, "suku_bunga": 7.0, "indeks_usd": 110.0})
    ]

    self.scenario_vars = {}
    for key, label, params in scenarios:
        var = tk.BooleanVar(value=True)
        self.scenario_vars[key] = var
        ttk.Checkbutton(scenario_frame, text=label, variable=var).pack(anchor=tk.W, padx=5, pady=1)

    ttk.Button(scenario_frame, text="📈 Analisis Skenario",
               command=self.jalankan_analisis_skenario).pack(pady=5)

    # Export options
    export_frame = ttk.LabelFrame(frame_kiri, text="Export Hasil")
    export_frame.pack(fill=tk.X, pady=5, padx=5)

    ttk.Button(export_frame, text="📄 Export Report",
               command=self.export_analisis_report).pack(fill=tk.X, padx=5, pady=2)
    ttk.Button(export_frame, text="📊 Export Charts",
               command=self.export_analisis_charts).pack(fill=tk.X, padx=5, pady=2)
    ttk.Button(export_frame, text="📋 Export Data",
               command=self.export_analisis_data).pack(fill=tk.X, padx=5, pady=2)

    # === Frame Kanan ===
    # Notebook untuk hasil analisis
    self.analisis_notebook = ttk.Notebook(frame_kanan)
    self.analisis_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Tab Korelasi
    self.tab_korelasi = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_korelasi, text="📊 Korelasi")
    self.setup_korelasi_tab()

    # Tab Sensitivitas
    self.tab_sensitivitas = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_sensitivitas, text="🎯 Sensitivitas")
    self.setup_sensitivitas_tab()

    # Tab Tren & Volatilitas
    self.tab_tren = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_tren, text="📈 Tren & Volatilitas")
    self.setup_tren_tab()

    # Tab Forecast
    self.tab_forecast = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_forecast, text="🔮 Forecast")
    self.setup_forecast_tab()

    # Tab Risk Analysis
    self.tab_risk = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_risk, text="⚠️ Risk Analysis")
    self.setup_risk_tab()

    # Tab Summary
    self.tab_summary = ttk.Frame(self.analisis_notebook)
    self.analisis_notebook.add(self.tab_summary, text="📋 Summary")
    self.setup_summary_tab()


def setup_korelasi_tab(self):
    """Setup correlation analysis tab"""
    self.fig_correlation = Figure(figsize=(12, 8))
    self.canvas_correlation = FigureCanvasTkAgg(self.fig_correlation, self.tab_korelasi)

    toolbar_frame = ttk.Frame(self.tab_korelasi)
    toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
    self.navbar_correlation = NavigationToolbar2Tk(self.canvas_correlation, toolbar_frame)

    self.canvas_correlation.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def setup_sensitivitas_tab(self):
    """Setup sensitivity analysis tab"""
    paned_sens = ttk.PanedWindow(self.tab_sensitivitas, orient=tk.VERTICAL)
    paned_sens.pack(fill=tk.BOTH, expand=True)

    plot_frame = ttk.Frame(paned_sens)
    paned_sens.add(plot_frame, weight=2)

    self.fig_sensitivity = Figure(figsize=(12, 6))
    self.canvas_sensitivity = FigureCanvasTkAgg(self.fig_sensitivity, plot_frame)

    toolbar_frame = ttk.Frame(plot_frame)
    toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
    self.navbar_sensitivity = NavigationToolbar2Tk(self.canvas_sensitivity, toolbar_frame)

    self.canvas_sensitivity.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    results_frame = ttk.LabelFrame(paned_sens, text="Sensitivity Results")
    paned_sens.add(results_frame, weight=1)

    self.sensitivity_text = tk.Text(results_frame, height=10)
    scroll_sens = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.sensitivity_text.yview)
    self.sensitivity_text.configure(yscrollcommand=scroll_sens.set)

    self.sensitivity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    scroll_sens.pack(side=tk.RIGHT, fill=tk.Y)


def setup_tren_tab(self):
    """Setup trend and volatility analysis tab"""
    self.fig_trend = Figure(figsize=(12, 10))
    self.canvas_trend = FigureCanvasTkAgg(self.fig_trend, self.tab_tren)

    toolbar_frame = ttk.Frame(self.tab_tren)
    toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
    self.navbar_trend = NavigationToolbar2Tk(self.canvas_trend, toolbar_frame)

    self.canvas_trend.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def setup_forecast_tab(self):
    """Setup forecast analysis tab"""
    paned_forecast = ttk.PanedWindow(self.tab_forecast, orient=tk.VERTICAL)
    paned_forecast.pack(fill=tk.BOTH, expand=True)

    plot_frame = ttk.Frame(paned_forecast)
    paned_forecast.add(plot_frame, weight=2)

    self.fig_forecast = Figure(figsize=(12, 6))
    self.canvas_forecast = FigureCanvasTkAgg(self.fig_forecast, plot_frame)

    toolbar_frame = ttk.Frame(plot_frame)
    toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
    self.navbar_forecast = NavigationToolbar2Tk(self.canvas_forecast, toolbar_frame)

    self.canvas_forecast.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    table_frame = ttk.LabelFrame(paned_forecast, text="Forecast Results")
    paned_forecast.add(table_frame, weight=1)

    self.forecast_tree = ttk.Treeview(table_frame,
                                      columns=("Year", "Predicted_Price", "Lower_CI", "Upper_CI", "Risk_Level"),
                                      show='headings', height=8)

    columns = [("Year", 80), ("Predicted_Price", 120), ("Lower_CI", 100), ("Upper_CI", 100), ("Risk_Level", 100)]
    for col, width in columns:
        self.forecast_tree.heading(col, text=col.replace("_", " "))
        self.forecast_tree.column(col, width=width, anchor=tk.CENTER)

    tree_scroll_v = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.forecast_tree.yview)
    tree_scroll_h = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.forecast_tree.xview)
    self.forecast_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)

    self.forecast_tree.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
    tree_scroll_v.grid(row=0, column=1, sticky='ns')
    tree_scroll_h.grid(row=1, column=0, sticky='ew')

    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)


def setup_risk_tab(self):
    """Setup risk analysis tab"""
    self.fig_risk = Figure(figsize=(12, 10))
    self.canvas_risk = FigureCanvasTkAgg(self.fig_risk, self.tab_risk)

    toolbar_frame = ttk.Frame(self.tab_risk)
    toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
    self.navbar_risk = NavigationToolbar2Tk(self.canvas_risk, toolbar_frame)

    self.canvas_risk.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def setup_summary_tab(self):
    """Setup summary tab"""
    self.summary_text = tk.Text(self.tab_summary, wrap=tk.WORD, font=("Courier", 10))
    summary_scroll = ttk.Scrollbar(self.tab_summary, orient=tk.VERTICAL, command=self.summary_text.yview)
    self.summary_text.configure(yscrollcommand=summary_scroll.set)

    self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)


def buat_tab_model_comparison(self):
    """Membuat tab perbandingan model"""
    bingkai = ttk.Frame(self.tab_model_comparison)
    bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    ttk.Label(bingkai, text="Tab Model Comparison - Akan diimplementasikan",
              font=("Arial", 14)).pack(pady=50)


def buat_tab_backtesting(self):
    """Membuat tab backtesting"""
    bingkai = ttk.Frame(self.tab_backtesting)
    bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    ttk.Label(bingkai, text="Tab Backtesting - Akan diimplementasikan",
              font=("Arial", 14)).pack(pady=50)


def buat_tab_teori(self):
    """Membuat tab teori dengan panduan yang diperluas"""
    bingkai = ttk.Frame(self.tab_teori)
    bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    teks_teori = """
        # 🔬 Teori Singular Value Decomposition (SVD)

        ## Pengantar
        SVD adalah teknik faktorisasi matriks yang sangat powerful untuk analisis data.
        Dalam konteks prediksi harga emas, SVD membantu kita:

        1. **Reduksi Dimensi**: Mengurangi kompleksitas data tanpa kehilangan informasi penting
        2. **Noise Reduction**: Menghilangkan noise dari data historis
        3. **Pattern Recognition**: Mengidentifikasi pola tersembunyi dalam data

        ## Formula Matematika

        Untuk matriks M (m×n), SVD memberikan:
        **M = UΣV^T**

        Dimana:
        - **U**: Matriks vektor singular kiri (m×m)
        - **Σ**: Matriks diagonal nilai singular (m×n)  
        - **V^T**: Transpose matriks vektor singular kanan (n×n)

        ## Implementasi dalam Prediksi Harga Emas

        ### Langkah 1: Persiapan Matriks
        ```
        M = [inflasi, suku_bunga, indeks_usd]
        y = [harga_emas]
        ```

        ### Langkah 2: Dekomposisi SVD
        ```
        U, Σ, V^T = SVD(M)
        ```

        ### Langkah 3: Menghitung Koefisien
        ```
        ω = V × Σ^+ × U^T × y
        ```

        ### Langkah 4: Prediksi
        ```
        y_pred = M_future × ω
        ```

        ## Keunggulan SVD
        - **Stabilitas Numerik**: Lebih stabil dibanding matrix inversion langsung
        - **Handling Multicollinearity**: Dapat menangani korelasi tinggi antar variabel
        - **Optimal Solution**: Memberikan solusi least squares yang optimal
        """

    scroll_teori = ttk.Scrollbar(bingkai)
    scroll_teori.pack(side=tk.RIGHT, fill=tk.Y)

    text_teori = tk.Text(bingkai, yscrollcommand=scroll_teori.set, wrap=tk.WORD,
                         font=("Arial", 10))
    text_teori.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    text_teori.insert(tk.END, teks_teori)
    text_teori.config(state=tk.DISABLED)

    scroll_teori.config(command=text_teori.yview)

    # === Utility Methods ===


def buat_tooltip(self, widget, text):
    """Membuat tooltip untuk widget"""

    def on_enter(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        label = ttk.Label(tooltip, text=text, background="lightyellow",
                          relief="solid", borderwidth=1)
        label.pack()
        widget.tooltip = tooltip

    def on_leave(event):
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
            del widget.tooltip

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def update_progress(self, value, status):
    """Update progress bar dan status"""
    self.progress_var.set(value)
    self.status_var.set(status)
    self.root.update_idletasks()


def reset_progress(self):
    """Reset progress bar"""
    self.progress_var.set(0)
    self.status_var.set("Siap")

    # === Core Prediction Methods ===


def perbarui_tampilan_data(self):
    """Memperbarui tampilan data di UI"""
    # Clear treeview
    for item in self.pohon_data.get_children():
        self.pohon_data.delete(item)

    # Add data to treeview
    for i in range(len(self.data_historis['tahun'])):
        self.pohon_data.insert("", "end", values=(
            self.data_historis['tahun'][i],
            self.data_historis['inflasi'][i],
            self.data_historis['suku_bunga'][i],
            self.data_historis['indeks_usd'][i],
            self.data_historis['harga_emas'][i]
        ))

    # Update statistics
    self.update_statistics()

    # Update visualizations
    self.perbarui_visualisasi_data()


def update_statistics(self):
    """Update statistik data"""
    if not self.data_historis['tahun']:
        return

    stats = []
    for key in ['inflasi', 'suku_bunga', 'indeks_usd', 'harga_emas']:
        data = np.array(self.data_historis[key])
        stats.append(f"{key.replace('_', ' ').title()}:")
        stats.append(f"  Mean: {np.mean(data):.2f}")
        stats.append(f"  Std: {np.std(data):.2f}")
        stats.append(f"  Min: {np.min(data):.2f}")
        stats.append(f"  Max: {np.max(data):.2f}")
        stats.append("")

    self.stats_text.config(state=tk.NORMAL)
    self.stats_text.delete(1.0, tk.END)
    self.stats_text.insert(tk.END, "\n".join(stats))
    self.stats_text.config(state=tk.DISABLED)


def perbarui_visualisasi_data(self, event=None):
    """Memperbarui visualisasi data"""
    if not self.data_historis['tahun']:
        return

    self.gambar_data.clear()

    jenis_plot = self.jenis_plot.get()
    var_selected = self.var_visualisasi.get()

    if var_selected == "semua":
        # Create subplots
        axes = []
        axes.append(self.gambar_data.add_subplot(221))
        axes.append(self.gambar_data.add_subplot(222))
        axes.append(self.gambar_data.add_subplot(223))
        axes.append(self.gambar_data.add_subplot(224))

        tahun = self.data_historis['tahun']
        data_vars = ['harga_emas', 'inflasi', 'suku_bunga', 'indeks_usd']
        colors = ['gold', 'red', 'blue', 'green']
        titles = ['Harga Emas (USD)', 'Inflasi (%)', 'Suku Bunga (%)', 'Indeks USD']

        for i, (var, color, title) in enumerate(zip(data_vars, colors, titles)):
            data = self.data_historis[var]

            if jenis_plot == "line":
                axes[i].plot(tahun, data, 'o-', color=color, linewidth=2, markersize=6)
            elif jenis_plot == "bar":
                axes[i].bar(tahun, data, color=color, alpha=0.7)
            elif jenis_plot == "scatter":
                axes[i].scatter(tahun, data, color=color, s=100)
            elif jenis_plot == "area":
                axes[i].fill_between(tahun, data, alpha=0.7, color=color)

            axes[i].set_title(title, fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].tick_params(axis='x', rotation=45)

    else:
        # Single variable plot
        ax = self.gambar_data.add_subplot(111)
        tahun = self.data_historis['tahun']
        data = self.data_historis[var_selected]

        if jenis_plot == "line":
            ax.plot(tahun, data, 'o-', linewidth=3, markersize=8)
        elif jenis_plot == "bar":
            ax.bar(tahun, data, alpha=0.8)
        elif jenis_plot == "scatter":
            ax.scatter(tahun, data, s=150)
        elif jenis_plot == "area":
            ax.fill_between(tahun, data, alpha=0.7)

        ax.set_title(f'{var_selected.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)

    self.gambar_data.tight_layout()
    self.kanvas_data.draw()


def jalankan_prediksi_async(self):
    """Jalankan prediksi secara asynchronous"""
    if self.sedang_memproses:
        return

    def prediksi_thread():
        self.sedang_memproses = True
        try:
            self.jalankan_prediksi()
        finally:
            self.sedang_memproses = False
            self.reset_progress()

    thread = threading.Thread(target=prediksi_thread)
    thread.daemon = True
    thread.start()


def jalankan_prediksi(self):
    """Menjalankan perhitungan prediksi dengan model yang dipilih"""
    try:
        self.update_progress(10, "Mempersiapkan data...")

        # Get parameters
        tahun = int(self.var_tahun.get())
        inflasi = float(self.var_inflasi.get())
        suku_bunga = float(self.var_suku_bunga.get())
        indeks_usd = float(self.var_usd.get())

        # Update prediction parameters
        self.parameter_prediksi.update({
            'tahun': tahun,
            'inflasi': inflasi,
            'suku_bunga': suku_bunga,
            'indeks_usd': indeks_usd
        })

        self.update_progress(30, "Mempersiapkan matriks...")

        # Prepare matrices
        M = np.array([
            self.data_historis['inflasi'],
            self.data_historis['suku_bunga'],
            self.data_historis['indeks_usd']
        ]).T

        y = np.array(self.data_historis['harga_emas'])
        Mx = np.array([inflasi, suku_bunga, indeks_usd])

        self.update_progress(50, "Menjalankan model...")

        # Get selected model
        model_type = self.model_terpilih.get()

        if model_type == "svd":
            harga_prediksi = self.prediksi_svd(M, y, Mx)
        else:
            harga_prediksi = self.prediksi_ml_model(M, y, Mx, model_type)

        self.update_progress(80, "Menghitung hasil...")

        # Store results
        self.hasil['harga_preimport numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import pandas as pd
        import csv
        import os
        import json
        from datetime import datetime, timedelta
        import io
        from PIL import Image, ImageTk
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import time
        from scipy import stats
        from scipy.signal import correlate
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import warnings

        warnings.filterwarnings('ignore')


class EnhancedAplikasiPrediksiHargaEmas:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Prediksi Harga Emas - Versi Ditingkatkan v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f5f5f5")

        # Konfigurasi tema
        self.tema_gelap = False
        self.warna_tema = {
            'terang': {
                'bg': '#f5f5f5',
                'fg': '#000000',
                'select_bg': '#0078d4',
                'select_fg': '#ffffff'
            },
            'gelap': {
                'bg': '#2d2d2d',
                'fg': '#ffffff',
                'select_bg': '#404040',
                'select_fg': '#ffffff'
            }
        }

        # Cache untuk optimasi
        self.cache_perhitungan = {}
        self.cache_visualisasi = {}

        # Status aplikasi
        self.sedang_memproses = False
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Siap")

        # Thread pool untuk operasi paralel
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Pengaturan aplikasi
        self.pengaturan = {
            'auto_save': True,
            'tema': 'terang',
            'bahasa': 'indonesia',
            'cache_enabled': True,
            'parallel_processing': True
        }

        try:
            self.root.iconbitmap("gold_icon.ico")
        except:
            pass

        # Data default dari makalah dengan data tambahan
        self.data_historis = {
            'tahun': [2020, 2021, 2022, 2023, 2024],
            'inflasi': [2.75, 3.12, 3.36, 1.80, 2.09],
            'suku_bunga': [3.5, 3.75, 4.0, 5.8125, 6.1042],
            'indeks_usd': [100.7, 95.6, 104.0558, 103.4642, 104.46],
            'harga_emas': [1773.3, 1807.2, 1806.9667, 1962.2, 2416.4217]
        }

        # Parameter prediksi (nilai default)
        self.parameter_prediksi = {
            'tahun': 2025,
            'inflasi': 3.5,
            'suku_bunga': 5.5,
            'indeks_usd': 108.0
        }

        # Penyimpanan hasil
        self.hasil = {
            'harga_prediksi': None,
            'komponen_svd': None,
            'vektor_omega': None,
            'perhitungan_manual': 3004.5476876,
            'persentase_galat': None,
            'analisis_sensitivitas': {},
            'confidence_interval': None,
            'model_comparison': {},
            'cross_validation_scores': None,
            'feature_importance': None,
            'residuals': None,
            'y_train_pred': None
        }

        # Model tambahan
        self.model_alternatif = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Membuat elemen UI utama
        self.buat_ui()
        self.buat_menu()
        self.buat_toolbar()
        self.buat_status_bar()

        # Memuat pengaturan
        self.muat_pengaturan()

        # Memuat data awal dan memperbarui UI
        self.perbarui_tampilan_data()

        # Keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Auto-save timer
        if self.pengaturan['auto_save']:
            self.setup_auto_save()

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.impor_data())
        self.root.bind('<Control-s>', lambda e: self.ekspor_hasil())
        self.root.bind('<Control-r>', lambda e: self.jalankan_prediksi())
        self.root.bind('<F5>', lambda e: self.refresh_data())
        self.root.bind('<Control-t>', lambda e: self.toggle_tema())

    def setup_auto_save(self):
        """Setup auto-save timer"""

        def auto_save():
            if self.pengaturan['auto_save']:
                self.simpan_pengaturan()
                self.root.after(300000, auto_save)  # 5 menit

        self.root.after(300000, auto_save)

    def buat_menu(self):
        """Membuat menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menu File
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Impor Data (Ctrl+O)", command=self.impor_data)
        file_menu.add_command(label="Ekspor Hasil (Ctrl+S)", command=self.ekspor_hasil)
        file_menu.add_separator()
        file_menu.add_command(label="Ekspor Laporan PDF", command=self.ekspor_laporan_pdf)
        file_menu.add_command(label="Ekspor Grafik", command=self.ekspor_grafik)
        file_menu.add_separator()
        file_menu.add_command(label="Keluar", command=self.root.quit)

        # Menu Edit
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Refresh Data (F5)", command=self.refresh_data)
        edit_menu.add_command(label="Reset Cache", command=self.reset_cache)
        edit_menu.add_separator()
        edit_menu.add_command(label="Pengaturan", command=self.buka_pengaturan)

        # Menu Analisis
        analisis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analisis", menu=analisis_menu)
        analisis_menu.add_command(label="Jalankan Prediksi (Ctrl+R)", command=self.jalankan_prediksi)
        analisis_menu.add_command(label="Analisis Komprehensif", command=self.analisis_komprehensif)
        analisis_menu.add_command(label="Backtesting", command=self.jalankan_backtesting)
        analisis_menu.add_command(label="Cross Validation", command=self.jalankan_cross_validation)

        # Menu Visualisasi
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualisasi", menu=viz_menu)
        viz_menu.add_command(label="Grafik Interaktif", command=self.buka_grafik_interaktif)
        viz_menu.add_command(label="Surface Plot 3D", command=self.buka_surface_plot_3d)
        viz_menu.add_command(label="Heatmap Korelasi", command=self.buka_heatmap_korelasi)

        # Menu Tema
        tema_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tema", menu=tema_menu)
        tema_menu.add_command(label="Tema Terang", command=lambda: self.ubah_tema('terang'))
        tema_menu.add_command(label="Tema Gelap", command=lambda: self.ubah_tema('gelap'))
        tema_menu.add_command(label="Toggle Tema (Ctrl+T)", command=self.toggle_tema)

        # Menu Bantuan
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bantuan", menu=help_menu)
        help_menu.add_command(label="Panduan Penggunaan", command=self.buka_panduan)
        help_menu.add_command(label="Tentang Aplikasi", command=self.buka_tentang)

    def buat_toolbar(self):
        """Membuat toolbar"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(fill=tk.X, padx=5, pady=2)

        # Tombol toolbar
        ttk.Button(self.toolbar, text="📁 Impor", command=self.impor_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="💾 Ekspor", command=self.ekspor_hasil).pack(side=tk.LEFT, padx=2)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(self.toolbar, text="🔍 Prediksi", command=self.jalankan_prediksi).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="📊 Analisis", command=self.analisis_komprehensif).pack(side=tk.LEFT, padx=2)
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(self.toolbar, text="🔄 Refresh", command=self.refresh_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="⚙️ Pengaturan", command=self.buka_pengaturan).pack(side=tk.LEFT, padx=2)

    def buat_status_bar(self):
        """Membuat status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Status label
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)

    def buat_ui(self):
        """Membuat antarmuka pengguna utama"""
        # Membuat bingkai utama
        self.buat_header()

        # Membuat kontainer utama
        kontainer_utama = ttk.Frame(self.root)
        kontainer_utama.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Menambahkan tab
        self.tab_control = ttk.Notebook(kontainer_utama)

        self.tab_data = ttk.Frame(self.tab_control)
        self.tab_prediksi = ttk.Frame(self.tab_control)
        self.tab_analisis = ttk.Frame(self.tab_control)
        self.tab_model_comparison = ttk.Frame(self.tab_control)
        self.tab_backtesting = ttk.Frame(self.tab_control)
        self.tab_teori = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_data, text='📊 Data Historis')
        self.tab_control.add(self.tab_prediksi, text='🔮 Prediksi')
        self.tab_control.add(self.tab_analisis, text='📈 Analisis Lanjutan')
        self.tab_control.add(self.tab_model_comparison, text='🔬 Perbandingan Model')
        self.tab_control.add(self.tab_backtesting, text='⏮️ Backtesting')
        self.tab_control.add(self.tab_teori, text='📚 Teori & Panduan')

        self.tab_control.pack(expand=1, fill=tk.BOTH)

        # Mengisi setiap tab dengan konten
        self.buat_tab_data()
        self.buat_tab_prediksi()
        self.buat_tab_analisis()
        self.buat_tab_model_comparison()
        self.buat_tab_backtesting()
        self.buat_tab_teori()

    def buat_header(self):
        """Membuat header aplikasi"""
        bingkai_header = ttk.Frame(self.root)
        bingkai_header.pack(fill=tk.X, pady=10)

        label_judul = ttk.Label(
            bingkai_header,
            text="🏆 Aplikasi Prediksi Harga Emas - Versi Ditingkatkan",
            font=("Arial", 18, "bold")
        )
        label_judul.pack()

        label_subjudul = ttk.Label(
            bingkai_header,
            text="Menggunakan SVD, Machine Learning & Analisis Statistik Lanjutan",
            font=("Arial", 12)
        )
        label_subjudul.pack()

    def buat_tab_data(self):
        """Membuat tab data historis dengan fitur enhanced"""
        bingkai = ttk.Frame(self.tab_data)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Data table dan kontrol
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri ===
        # Data controls
        kontrol_frame = ttk.LabelFrame(frame_kiri, text="Kontrol Data")
        kontrol_frame.pack(fill=tk.X, pady=5)

        # Tombol kontrol dalam grid
        ttk.Button(kontrol_frame, text="➕ Tambah", command=self.tambah_baris_data).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="✏️ Edit", command=self.edit_baris_data).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="🗑️ Hapus", command=self.hapus_baris_data).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="📥 Impor", command=self.impor_data).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="📤 Ekspor", command=self.ekspor_data).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(kontrol_frame, text="🔄 Refresh", command=self.refresh_data).grid(row=1, column=2, padx=2, pady=2)

        # Tabel data dengan scrollbar
        table_frame = ttk.LabelFrame(frame_kiri, text="Data Historis")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Treeview dengan scrollbar
        tree_frame = ttk.Frame(table_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pohon_data = ttk.Treeview(
            tree_frame,
            columns=("Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD", "Harga Emas (USD)"),
            show='headings'
        )

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.pohon_data.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.pohon_data.xview)
        self.pohon_data.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout
        self.pohon_data.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Setup kolom
        for col in ("Tahun", "Inflasi (%)", "Suku Bunga (%)", "Indeks USD", "Harga Emas (USD)"):
            self.pohon_data.heading(col, text=col, command=lambda c=col: self.sort_treeview(c))
            self.pohon_data.column(col, width=100, anchor=tk.CENTER)

        # Statistik data
        stats_frame = ttk.LabelFrame(frame_kiri, text="Statistik Data")
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_text = tk.Text(stats_frame, height=8, width=30)
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan ===
        # Kontrol visualisasi
        viz_control_frame = ttk.LabelFrame(frame_kanan, text="Kontrol Visualisasi")
        viz_control_frame.pack(fill=tk.X, pady=5)

        # Row 1
        row1 = ttk.Frame(viz_control_frame)
        row1.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(row1, text="Jenis Grafik:").pack(side=tk.LEFT, padx=5)
        self.jenis_plot = tk.StringVar(value="line")
        combo_plot = ttk.Combobox(row1, textvariable=self.jenis_plot,
                                  values=["line", "bar", "scatter", "area"], width=10, state="readonly")
        combo_plot.pack(side=tk.LEFT, padx=5)
        combo_plot.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_data)

        ttk.Label(row1, text="Variabel:").pack(side=tk.LEFT, padx=5)
        self.var_visualisasi = tk.StringVar(value="semua")
        combo_var = ttk.Combobox(row1, textvariable=self.var_visualisasi,
                                 values=["semua", "harga_emas", "inflasi", "suku_bunga", "indeks_usd"],
                                 width=12, state="readonly")
        combo_var.pack(side=tk.LEFT, padx=5)
        combo_var.bind("<<ComboboxSelected>>", self.perbarui_visualisasi_data)

        # Row 2
        row2 = ttk.Frame(viz_control_frame)
        row2.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(row2, text="🔄 Perbarui", command=self.perbarui_visualisasi_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="💾 Simpan Grafik", command=self.simpan_grafik_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(row2, text="🔍 Zoom Reset", command=self.reset_zoom_data).pack(side=tk.LEFT, padx=5)

        # Area visualisasi dengan toolbar
        viz_frame = ttk.LabelFrame(frame_kanan, text="Visualisasi Data")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.gambar_data = Figure(figsize=(10, 8), dpi=100)
        self.kanvas_data = FigureCanvasTkAgg(self.gambar_data, viz_frame)

        # Toolbar navigasi matplotlib
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=2)

        self.navbar_data = NavigationToolbar2Tk(self.kanvas_data, toolbar_frame)
        self.navbar_data.update()

        self.kanvas_data.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def buat_tab_prediksi(self):
        """Membuat tab prediksi dengan fitur enhanced"""
        bingkai = ttk.Frame(self.tab_prediksi)
        bingkai.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # PanedWindow untuk split layout
        paned = ttk.PanedWindow(bingkai, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Frame kiri - Parameter dan kontrol
        frame_kiri = ttk.Frame(paned)
        paned.add(frame_kiri, weight=1)

        # Frame kanan - Hasil dan visualisasi
        frame_kanan = ttk.Frame(paned)
        paned.add(frame_kanan, weight=2)

        # === Frame Kiri ===
        # Parameter prediksi dengan tooltips
        param_frame = ttk.LabelFrame(frame_kiri, text="Parameter Prediksi")
        param_frame.pack(fill=tk.X, pady=5, padx=5)

        # Tahun
        ttk.Label(param_frame, text="Tahun:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_tahun = tk.StringVar(value=str(self.parameter_prediksi['tahun']))
        entry_tahun = ttk.Entry(param_frame, textvariable=self.var_tahun, width=10)
        entry_tahun.grid(row=0, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_tahun, "Masukkan tahun untuk prediksi (contoh: 2025)")

        # Inflasi
        ttk.Label(param_frame, text="Inflasi (%):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_inflasi = tk.StringVar(value=str(self.parameter_prediksi['inflasi']))
        entry_inflasi = ttk.Entry(param_frame, textvariable=self.var_inflasi, width=10)
        entry_inflasi.grid(row=1, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_inflasi, "Tingkat inflasi yang diharapkan dalam persen")

        # Suku Bunga
        ttk.Label(param_frame, text="Suku Bunga (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_suku_bunga = tk.StringVar(value=str(self.parameter_prediksi['suku_bunga']))
        entry_suku_bunga = ttk.Entry(param_frame, textvariable=self.var_suku_bunga, width=10)
        entry_suku_bunga.grid(row=2, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_suku_bunga, "Suku bunga acuan dalam persen")

        # Indeks USD
        ttk.Label(param_frame, text="Indeks USD:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.var_usd = tk.StringVar(value=str(self.parameter_prediksi['indeks_usd']))
        entry_usd = ttk.Entry(param_frame, textvariable=self.var_usd, width=10)
        entry_usd.grid(row=3, column=1, padx=5, pady=5)
        self.buat_tooltip(entry_usd, "Indeks kekuatan Dollar AS")

        # Tombol prediksi
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Button(btn_frame, text="🔮 Hitung Prediksi",
                   command=self.jalankan_prediksi_async).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📊 Analisis Komprehensif",
                   command=self.analisis_komprehensif).pack(side=tk.LEFT, padx=2)

        # Model selection
        model_frame = ttk.LabelFrame(frame_kiri, text="Pilihan Model")
        model_frame.pack(fill=tk.X, pady=5, padx=5)

        self.model_terpilih = tk.StringVar(value="svd")
        models = [
            ("SVD + Least Squares", "svd"),
            ("Linear Regression", "linear"),
            ("Ridge Regression", "ridge"),
            ("Lasso Regression", "lasso"),
            ("Random Forest", "random_forest")
        ]

        for i, (text, value) in enumerate(models):
            ttk.Radiobutton(model_frame, text=text, variable=self.model_terpilih,
                            value=value).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

        # Opsi lanjutan
        advanced_frame = ttk.LabelFrame(frame_kiri, text="Opsi Lanjutan")
        advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        self.var_confidence = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Hitung Confidence Interval",
                        variable=self.var_confidence).pack(anchor=tk.W, padx=5, pady=2)

        self.var_feature_importance = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Analisis Feature Importance",
                        variable=self.var_feature_importance).pack(anchor=tk.W, padx=5, pady=2)

        self.var_residual_analysis = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Analisis Residual",
                        variable=self.var_residual_analysis).pack(anchor=tk.W, padx=5, pady=2)

        # SVD Components
        svd_frame = ttk.LabelFrame(frame_kiri, text="Komponen SVD")
        svd_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        self.teks_svd = tk.Text(svd_frame, height=12, width=40)
        svd_scroll = ttk.Scrollbar(svd_frame, orient=tk.VERTICAL, command=self.teks_svd.yview)
        self.teks_svd.configure(yscrollcommand=svd_scroll.set)

        self.teks_svd.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        svd_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # === Frame Kanan ===
        # Hasil prediksi
        hasil_frame = ttk.LabelFrame(frame_kanan, text="Hasil Prediksi")
        hasil_frame.pack(fill=tk.X, pady=5, padx=5)

        # Hasil utama dengan styling
        self.label_hasil = ttk.Label(hasil_frame, text="Jalankan prediksi untuk melihat hasil",
                                     font=("Arial", 24, "bold"), foreground="blue")
        self.label_hasil.pack(pady=10)

        # Detail hasil
        detail_notebook = ttk.Notebook(hasil_frame)
        detail_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab detail umum
        tab_detail = ttk.Frame(detail_notebook)
        detail_notebook.add(tab_detail, text="Detail")

        self.teks_detail = tk.Text(tab_detail, height=8, width=60)
        detail_scroll = ttk.Scrollbar(tab_detail, orient=tk.VERTICAL, comman