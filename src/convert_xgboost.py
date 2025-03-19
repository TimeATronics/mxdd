"""
    This file is part of the MXDD distribution (https://github.com/TimeATronics/mxdd).
    Copyright (c) 2025 Aradhya Chakrabarti

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but 
    WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
    General Public License for more details.

    You should have received a copy of the GNU General Public License 
    along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import onnxmltools
import joblib
xgb_model = joblib.load("mnet_xgboost.pkl")
initial_type = [("float_input", onnxmltools.convert.common.data_types.FloatTensorType([None, 1280]))]
onnx_xgb = onnxmltools.convert.convert_xgboost(xgb_model, initial_types=initial_type)
onnxmltools.utils.save_model(onnx_xgb, "xgboost_model.onnx")
print("XGBoost model saved as xgboost_model.onnx")
